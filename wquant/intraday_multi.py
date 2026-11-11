"""夜盘多时点 + 短期趋势偏向 + 可持仓过夜 的回测引擎（SOXL / SOXS）。

和 intraday24 的区别：
- 夜盘里有多个检查时点（例如 21:00 / 23:00 / 01:30 韩国收盘 / 04:00），空仓时第一次满足信号就进场
- 可以加"短期趋势偏向"（bias）：只做和偏向一致的方向
- carry=True 时，偏向和持仓同方向就持有过夜；第二天检查时点出现反向信号、偏向改变或止损才离场
- 仓位按当前权益的比例（复利），用于评估小资金放大仓位的效果
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PM:
    checkpoints: tuple = ("04:00",)       # 美东时间；20:00-23:59 属于前一天晚上
    threshold: float = 0.03
    korea_confirm: float | None = 0.01
    stop_pct: float = 0.06                # 固定止损（持有标的）
    trail_pct: float | None = None        # 移动止损
    carry: bool = False                   # 偏向一致时持仓过夜
    frac: float = 0.33                    # 每笔仓位 = 权益 × frac
    capital: float = 3_000
    slip_rth: float = 0.0005
    slip_ext: float = 0.0015
    reentry: bool = True
    rebound_pct: float = 0.03
    reentry_mode: str = "rebound"         # rebound = 从低点反弹 rebound_pct；new_high = 涨回卖出前持仓最高点之上
    max_reentries: int | None = 1         # 一天最多买回几次；None = 不限
    daily_loss_stop: float | None = None  # 当天亏损超过权益的这个比例，当天不再开仓
    # 融资：夜盘不能融资，仓位最多 max_frac_ovn；盘前 / 常规 / 盘后最多 max_frac
    max_frac_ovn: float = 1.0
    max_frac: float = 1.0
    # 盈利加仓：盘前或常规时段，持有标的比上次买入价再涨 add_step，就在下一根 K 线加 add_frac，直到 max_frac
    add_step: float | None = None
    add_frac: float = 0.33
    add_last: str = "15:00"
    add_sessions: tuple = ("PRE", "RTH")
    trail_after_add: float | None = None  # 加过仓之后，整个仓位按最高点回撤这个比例止损
    # 隔夜持仓：carry_rule = None（每天平仓，旧逻辑）/ always / profit（收盘有浮盈才留）/ trend（SOXL 当天同方向 ≥ carry_min 才留）
    carry_rule: str | None = None
    carry_min: float = 0.0
    carry_weekend: bool = True            # 周五收盘是否也留仓过周末
    carry_korea_exit: float | None = None # 隔夜仓第二天在检查时点看韩国：KOSPI 反方向 ≥ 这个比例就离场
    eod: dict | None = None               # 尾盘开仓并隔夜 {enabled, at, threshold, mode: trend/fade}
    rth_exit: dict | None = None          # 常规时段走势转弱离场 {mode: vwap/orb, buffer, from, until, flip, max_flips, frac}
    fade: dict | None = None              # 冲高回落反手 {enabled, drop, min_run, from, to, vwap, only_untraded, frac}
    rth_entry: dict | None = None         # 常规时段顺势开仓 {enabled, from, to, threshold, vwap, korea, frac}
    path: dict | None = None              # 分时路径判断，键同 swing_live DEFAULTS["tactical"]["path"]


def _slip(sess, p):
    return p.slip_rth if sess == "RTH" else p.slip_ext


def _cp_times(day, cps, tz):
    out = []
    for c in cps:
        h, m = map(int, c.split(":"))
        t = pd.Timestamp(day).tz_localize(tz) + pd.Timedelta(hours=h, minutes=m)
        if h >= 16:
            t -= pd.Timedelta(days=1)
        out.append(t)
    return sorted(out)


def run_multi(L: pd.DataFrame, S: pd.DataFrame, K: pd.DataFrame | None, p: PM,
              bias: dict | None = None, size_fn=None) -> dict:
    """size_fn(day, side, soxl_move, kospi_move) -> 仓位比例；None 时用固定 p.frac。"""
    """bias: {trading_day(date): +1 / -1 / 0}，0 = 两边都可以；None = 不用偏向。"""
    tz = L.index.tz
    rthL = L[L.session == "RTH"]
    days = sorted(set(rthL.index.date))
    closeL = rthL.groupby(rthL.index.date).close.last()
    closeK = None
    if K is not None and "session" in K:
        rk = K[K.session == "RTH"]
        closeK = rk.groupby(rk.index.date).close.last()
    inst = {1: L, -1: S}
    equity = p.capital
    pos = None            # dict(side, qty, entry, peak, t_entry)
    trades, daily_eq = [], {}
    stopped_today = None
    for prev, d in zip(days, days[1:]):
        t0 = rthL[rthL.index.date == prev].index[-1]
        t1 = rthL[rthL.index.date == d].index[-1]
        prev_close = closeL[prev]
        kprev = closeK.get(prev) if closeK is not None else None
        _pi = days.index(prev)
        soxl_prev_ret = (prev_close / closeL[days[_pi - 1]] - 1) if _pi > 0 else 0.0
        cps = _cp_times(d, p.checkpoints, tz)
        b = (bias or {}).get(d, 0) if bias is not None else 0
        day_start_eq = equity
        halted = False
        low_after_stop, reentered = None, 0
        peak_before_stop = None
        win = L[(L.index > t0) & (L.index <= t1)]

        def mark(side):
            X = inst[side]
            return X

        def close_pos(t, px, reason):
            nonlocal pos, equity
            pnl = (px - pos["entry"]) * pos["qty"]
            equity += pnl
            trades.append({**pos, "t_exit": t, "exit": px, "pnl": pnl, "reason": reason, "day": d})
            pos = None
            return pnl

        day_frac = {"f": p.frac}

        def open_pos(side, t_next, frac=None):
            nonlocal pos
            X = inst[side]
            i = X.index.searchsorted(t_next)
            if i >= len(X) or X.index[i] > t1:
                return
            px = X.open.iloc[i] * (1 + _slip(X.session.iloc[i], p))
            f = frac if frac is not None else day_frac["f"]
            f = min(f, p.max_frac_ovn if X.session.iloc[i] == "OVN" else p.max_frac)
            qty = equity * f / px
            pos = {"side": side, "qty": qty, "entry": px, "peak": px, "t_entry": X.index[i], "frac": f,
                   "last_add": px, "adds": 0, "pending_add": False}

        # 偏向变了：持仓方向和今天的偏向相反，就在第一个检查时点离场（下面的循环里处理）
        add_cut = pd.Timestamp(f"{d} {p.add_last}").tz_localize(tz)
        rth_used = False
        _rc = p.rth_entry or {}
        rth_lo = pd.Timestamp(f"{d} {_rc.get('from', '10:00')}").tz_localize(tz)
        rth_hi = pd.Timestamp(f"{d} {_rc.get('to', '12:00')}").tz_localize(tz)
        n_trades_start = len(trades)

        def traded_today():
            return len(trades) > n_trades_start or pos is not None
        _pc = p.path or {}
        scan_lo = _cp_times(d, (_pc["scan_from"],), tz)[0] if _pc.get("scan_from") else cps[0]
        scan_hi = _cp_times(d, (_pc["scan_to"],), tz)[0] if _pc.get("scan_to") else cps[-1]
        # 常规时段的 VWAP 和开盘区间（只用到当前 K 线为止的数据）
        _rth = win[win.session == "RTH"]
        if len(_rth):
            _tp = (_rth.high + _rth.low + _rth.close) / 3
            rth_vwap = (_tp * _rth.volume).cumsum() / _rth.volume.cumsum().replace(0, np.nan)
            or_end = _rth.index[0] + pd.Timedelta(minutes=30)
            _or = _rth[_rth.index < or_end]
            or_hi, or_lo = _or.high.max(), _or.low.min()
        else:
            rth_vwap, or_end, or_hi, or_lo = pd.Series(dtype=float), None, None, None
        flips = 0
        fade_used = False
        kswing_used = False
        kclose_t = _cp_times(d, ("01:30",), tz)[0]
        eod_t = pd.Timestamp(f"{d} {(p.eod or {}).get('at', '15:30')}").tz_localize(tz)
        xc = p.rth_exit or {}
        fc = p.fade or {}
        x_from = pd.Timestamp(f"{d} {xc.get('from', '10:00')}").tz_localize(tz)
        x_until = pd.Timestamp(f"{d} {xc.get('until', '15:00')}").tz_localize(tz)
        f_from = pd.Timestamp(f"{d} {fc.get('from', '10:00')}").tz_localize(tz)
        f_to = pd.Timestamp(f"{d} {fc.get('to', '14:00')}").tz_localize(tz)
        for t in win.index:
            # ---- 执行上一根 K 线触发的加仓（本根开盘成交）
            if pos is not None and pos["pending_add"]:
                X = inst[pos["side"]]
                pos["pending_add"] = False
                if t in X.index and X.loc[t].session != "OVN":
                    px = X.loc[t].open * (1 + _slip(X.loc[t].session, p))
                    f_add = min(p.add_frac, p.max_frac - pos["frac"])
                    if f_add > 1e-6:
                        q_add = equity * f_add / px
                        pos["entry"] = (pos["entry"] * pos["qty"] + px * q_add) / (pos["qty"] + q_add)
                        pos["qty"] += q_add
                        pos["frac"] += f_add
                        pos["last_add"] = px
                        pos["adds"] += 1
            # ---- 持仓：止损检查（用持有标的这一根 K 线）
            if pos is not None:
                X = inst[pos["side"]]
                if t in X.index:
                    bar = X.loc[t]
                    stops = [pos["entry"] * (1 - p.stop_pct)]
                    if p.trail_pct:
                        stops.append(pos["peak"] * (1 - p.trail_pct))
                    if p.trail_after_add and pos["adds"]:
                        stops.append(pos["peak"] * (1 - p.trail_after_add))
                    st = max(stops)
                    if bar.low <= st:
                        peak_before_stop = pos["peak"]
                        close_pos(t, min(bar.open, st) * (1 - _slip(bar.session, p)), "止损")
                        low_after_stop = bar.low
                        stopped_side = trades[-1]["side"]
                        if p.daily_loss_stop and equity <= day_start_eq * (1 - p.daily_loss_stop):
                            halted = True
                    else:
                        pos["peak"] = max(pos["peak"], bar.high)
                        # ---- 盈利加仓的触发（这根收盘满足条件，下一根开盘加）
                        if (p.add_step and bar.session in p.add_sessions and t < add_cut
                                and pos["frac"] < p.max_frac - 1e-6 and bar.close >= pos["last_add"] * (1 + p.add_step)):
                            pos["pending_add"] = True
            # ---- 止损后买回
            if (pos is None and low_after_stop is not None and p.reentry and not halted
                    and (p.max_reentries is None or reentered < p.max_reentries)):
                X = inst[stopped_side]
                if t in X.index and t.hour < 15:
                    low_after_stop = min(low_after_stop, X.loc[t].low)
                    c = X.loc[t].close
                    hit = (c >= peak_before_stop) if p.reentry_mode == "new_high" else (c >= low_after_stop * (1 + p.rebound_pct))
                    if hit:
                        reentered += 1
                        open_pos(stopped_side, t + pd.Timedelta(minutes=5))
            pc = p.path or {}
            # ---- 隔夜仓：第二天检查时点 KOSPI 明显反方向就离场
            if (pos is not None and pos.get("carried") and p.carry_korea_exit is not None and t in cps
                    and K is not None and "prev_close" in K):
                from .intraday_data import kospi_move
                km = kospi_move(K, t)
                if km is not None:
                    if pc.get("korea_beta"):
                        km -= pc["korea_beta"] * soxl_prev_ret
                    if km * pos["side"] <= -p.carry_korea_exit:
                        X = inst[pos["side"]]
                        i = X.index.searchsorted(t + pd.Timedelta(minutes=5))
                        if i < len(X):
                            close_pos(X.index[i], X.open.iloc[i] * (1 - _slip(X.session.iloc[i], p)), "隔夜仓韩国不确认")
            # ---- 尾盘开仓（之后隔夜）
            ec = p.eod or {}
            if ec.get("enabled") and pos is None and not halted and t == eod_t:
                mv_e = win.close.loc[t] / prev_close - 1
                se = 1 if mv_e >= ec["threshold"] else -1 if mv_e <= -ec["threshold"] else 0
                if ec.get("mode") == "fade":
                    se = -se
                if se and (not b or se == b):
                    open_pos(se, t + pd.Timedelta(minutes=5), frac=ec.get("frac", p.frac))
                    if pos is not None:
                        pos["eod"] = True
            # ---- 检查时点：入场 / 反向
            scan_now = (t in cps) or (pc.get("enabled") and pc.get("scan") == "continuous" and pos is None
                                      and low_after_stop is None and scan_lo <= t <= scan_hi)
            if scan_now and not halted:
                px_t = win.close.loc[t]
                mv = px_t / prev_close - 1
                anchor = pc.get("anchor", "prev_close") if pc.get("enabled") else "prev_close"
                if anchor != "prev_close":
                    from .swing_live import anchor_sides, pick_side
                    from .intraday_data import kospi_path
                    w = win[win.index <= t]
                    lo, hi = w.low.min(), w.high.max()
                    cands = anchor_sides(mv, px_t / lo - 1, px_t / hi - 1, p.threshold, anchor)
                    sig = pick_side(cands, w.low.idxmin() > w.high.idxmax())
                    if sig and pc.get("veto_pct"):
                        if (sig < 0 and px_t / lo - 1 >= pc["veto_pct"]) or (sig > 0 and px_t / hi - 1 <= -pc["veto_pct"]):
                            sig = 0
                    s_strength = cands.get(sig, 0.0)
                    kmv = None
                    if sig and p.korea_confirm is not None:
                        kp = kospi_path(K, t) if K is not None else None
                        kc = anchor_sides(*kp, p.korea_confirm, pc.get("korea_anchor", anchor)) if kp else {}
                        kmv = kc.get(sig)
                        if kmv is None:
                            sig = 0
                    if sig and b and sig != b:
                        sig = 0
                    if pos is not None and ((sig and sig != pos["side"]) or (b and b != pos["side"])):
                        X = inst[pos["side"]]
                        i = X.index.searchsorted(t + pd.Timedelta(minutes=5))
                        if i < len(X):
                            close_pos(X.index[i], X.open.iloc[i] * (1 - _slip(X.session.iloc[i], p)), "反向信号 / 偏向改变")
                    if pos is None and sig and low_after_stop is None:
                        if size_fn is not None:
                            day_frac["f"] = float(size_fn(d, sig, sig * s_strength, sig * (kmv or 0.0)))
                        if day_frac["f"] > 0:
                            open_pos(sig, t + pd.Timedelta(minutes=5))
                    continue
                if pc.get("enabled"):
                    w = win[win.index <= t]
                    vol = w.volume.sum()
                    vwap = float(((w.high + w.low + w.close) / 3 * w.volume).sum() / vol) if vol > 0 else None
                    from .swing_live import path_signal
                    sig, _, is_rev = path_signal(mv, px_t, w.low.min(), w.high.max(), vwap, p.threshold, pc)
                else:
                    sig, is_rev = (1 if mv >= p.threshold else -1 if mv <= -p.threshold else 0), False
                kmv = None
                korea_needed = not is_rev or pc.get("reversal_korea", True)
                if sig and korea_needed and p.korea_confirm is not None and K is not None and ("prev_close" in K or kprev is not None):
                    if "prev_close" in K:            # KOSPI 指数：自带上一交易日收盘
                        from .intraday_data import kospi_move
                        kmv = kospi_move(K, t) or 0.0
                        if pc.get("korea_beta"):     # 减去前一天美股带来的惯性，只看韩国自己的新信息
                            kmv -= pc["korea_beta"] * soxl_prev_ret
                    else:                            # EWY 等美股 ETF：相对其美股昨收
                        ki = K.index.searchsorted(t, side="right") - 1
                        kmv = K.close.iloc[ki] / kprev - 1 if ki >= 0 else 0
                    kc = p.korea_confirm
                    # SOXL 走得够猛时放宽韩国确认（可选：只在韩国收盘之后）
                    sm = pc.get("strong_move")
                    if sm and abs(mv) >= sm and (not pc.get("strong_after_close") or t >= _cp_times(d, ("01:30",), tz)[0]):
                        kc = pc.get("strong_korea", kc)          # None = 不看韩国；0 = 只要同向
                    if kc is None:
                        pass
                    elif (kc > 0 and (np.sign(kmv) != sig or abs(kmv) < kc)) or (kc == 0 and np.sign(kmv) == -sig):
                        sig = 0
                if sig and b and sig != b:
                    sig = 0
                if pos is not None and ((sig and sig != pos["side"]) or (b and b != pos["side"])):
                    X = inst[pos["side"]]
                    i = X.index.searchsorted(t + pd.Timedelta(minutes=5))
                    if i < len(X):
                        close_pos(X.index[i], X.open.iloc[i] * (1 - _slip(X.session.iloc[i], p)), "反向信号 / 偏向改变")
                if pos is None and sig and low_after_stop is None:
                    if size_fn is not None:
                        day_frac["f"] = float(size_fn(d, sig, mv, kmv))
                    if day_frac["f"] > 0:
                        open_pos(sig, t + pd.Timedelta(minutes=5))
            # ---- 韩国收盘后看 KOSPI 当天的回落 / 反弹：KOSPI 离当天高点跌 ≥ kdrop 且 SOXL 跌够 → 做空；反之做多
            kq = pc.get("kswing") or {}
            if (kq and pos is None and low_after_stop is None and not halted and K is not None and "prev_close" in K
                    and kclose_t <= t <= scan_hi and not kswing_used):
                from .intraday_data import kospi_path
                kp = kospi_path(K, t)
                if kp is not None:
                    _, k_up, k_dn = kp
                    w = win[win.index <= t]
                    c = w.close.iloc[-1]
                    if kq.get("soxl_anchor", "prev") == "prev":
                        s_dn = s_up = c / prev_close - 1
                    else:
                        s_dn, s_up = c / w.high.max() - 1, c / w.low.min() - 1
                    sig = 0
                    if k_dn <= -kq["kdrop"] and s_dn <= -kq["soxl"]:
                        sig = -1
                    elif k_up >= kq["kdrop"] and s_up >= kq["soxl"]:
                        sig = 1
                    if sig and pc.get("veto_pct"):           # 已经从低点反弹 6% 不追空（反之亦然）
                        if (sig < 0 and c / w.low.min() - 1 >= pc["veto_pct"]) or (sig > 0 and c / w.high.max() - 1 <= -pc["veto_pct"]):
                            sig = 0
                    if sig and b and sig != b:
                        sig = 0
                    if sig:
                        kswing_used = True
                        open_pos(sig, t + pd.Timedelta(minutes=5), frac=kq.get("frac", p.frac))
                        if pos is not None:
                            pos["kswing"] = True
            # ---- 常规时段顺势开仓：当天还没做过（夜盘没进场），开盘后 SOXL 相对昨收超过阈值就顺势买
            rc = p.rth_entry or {}
            if (rc.get("enabled") and pos is None and low_after_stop is None and not halted and not rth_used
                    and rth_lo <= t <= rth_hi and win.session.loc[t] == "RTH" and not traded_today()):
                px_t = win.close.loc[t]
                mv = px_t / prev_close - 1
                sig = 1 if mv >= rc["threshold"] else -1 if mv <= -rc["threshold"] else 0
                if sig and rc.get("vwap"):
                    w = win[(win.index <= t) & (win.session == "RTH")]
                    vol = w.volume.sum()
                    if vol > 0:
                        vw = ((w.high + w.low + w.close) / 3 * w.volume).sum() / vol
                        if (sig > 0 and px_t < vw) or (sig < 0 and px_t > vw):
                            sig = 0
                if sig and rc.get("korea") is not None and K is not None and "prev_close" in K:
                    from .intraday_data import kospi_move
                    kmv = kospi_move(K, t) or 0.0
                    if np.sign(kmv) != sig or abs(kmv) < rc["korea"]:
                        sig = 0
                if sig and b and sig != b:
                    sig = 0
                if sig:
                    rth_used = True
                    open_pos(sig, t + pd.Timedelta(minutes=5), frac=rc.get("frac", p.frac))
                    if pos is not None:
                        pos["rth"] = True
            # ---- 常规时段走势转弱 → 离场（可选反手）。用 SOXL 这根 K 线收盘判断，下一根开盘成交
            if (xc.get("mode") and pos is not None and t in rth_vwap.index and t >= x_from
                    and (xc["mode"] != "orb" or t >= or_end)):
                c = win.close.loc[t]
                buf = xc.get("buffer", 0.0)
                if xc["mode"] == "vwap":
                    ref_dn, ref_up = rth_vwap.loc[t], rth_vwap.loc[t]
                else:
                    ref_dn, ref_up = or_lo, or_hi
                weak = (pos["side"] > 0 and c < ref_dn * (1 - buf)) or (pos["side"] < 0 and c > ref_up * (1 + buf))
                if weak:
                    side_was = pos["side"]
                    X = inst[side_was]
                    i = X.index.searchsorted(t + pd.Timedelta(minutes=5))
                    if i < len(X) and X.index[i] <= t1:
                        close_pos(X.index[i], X.open.iloc[i] * (1 - _slip(X.session.iloc[i], p)),
                                  "跌破 VWAP" if xc["mode"] == "vwap" else "跌破开盘区间")
                        low_after_stop = None
                        if xc.get("flip") and flips < xc.get("max_flips", 1) and t < x_until:
                            flips += 1
                            open_pos(-side_was, t + pd.Timedelta(minutes=5), frac=xc.get("frac", p.frac))
                            if pos is not None:
                                pos["flip"] = True
                        else:
                            halted = True          # 当天不再开新仓
            # ---- 冲高回落 / 探底回升反手：当天还没仓位，常规时段从昨收以来的高点回落 ≥ drop 且在 VWAP 另一侧
            if (fc.get("enabled") and pos is None and not halted and not fade_used and f_from <= t <= f_to
                    and t in rth_vwap.index and (not fc.get("only_untraded", True) or not traded_today())):
                w = win[win.index <= t]
                c = w.close.iloc[-1]
                hi, lo = w.high.max(), w.low.min()
                v = rth_vwap.loc[t]
                sig = 0
                if c / hi - 1 <= -fc["drop"] and hi / prev_close - 1 >= fc.get("min_run", 0.0) and (not fc.get("vwap", True) or c < v):
                    sig = -1
                elif c / lo - 1 >= fc["drop"] and lo / prev_close - 1 <= -fc.get("min_run", 0.0) and (not fc.get("vwap", True) or c > v):
                    sig = 1
                if sig:
                    fade_used = True
                    open_pos(sig, t + pd.Timedelta(minutes=5), frac=fc.get("frac", p.frac))
                    if pos is not None:
                        pos["fade"] = True
        # ---- 当天收盘：不持仓过夜，或偏向不一致 → 平仓
        if pos is not None:
            X = inst[pos["side"]]
            keep = p.carry and b == pos["side"]
            if p.carry_rule or pos.get("eod"):
                X_last = X[X.index <= t1]
                cpx = X_last.close.iloc[-1]
                day_mv = closeL[d] / prev_close - 1
                rule = p.carry_rule
                keep = bool(rule) and (rule == "always" or (rule == "profit" and cpx > pos["entry"])
                        or (rule == "trend" and day_mv * pos["side"] >= p.carry_min))
                if pos.get("eod") and pos["t_entry"].date() == d:
                    keep = True                       # 尾盘刚开的仓一定留到第二天
                if b and b != pos["side"]:
                    keep = False
                if not p.carry_weekend and d.weekday() == 4:
                    keep = False
                if keep:
                    pos["carried"] = True
            if not keep:
                last = X[X.index <= t1]
                close_pos(last.index[-1], last.close.iloc[-1] * (1 - _slip("RTH", p)), "收盘平仓")
        # 当天收盘权益（持仓按收盘价盯市）
        mv = 0.0
        if pos is not None:
            X = inst[pos["side"]]
            last = X[X.index <= t1]
            mv = (last.close.iloc[-1] - pos["entry"]) * pos["qty"]
        daily_eq[d] = equity + mv
    eq = pd.Series(daily_eq).sort_index()
    return {"equity": eq, "trades": pd.DataFrame(trades)}


def summary(res: dict, p: PM) -> dict:
    eq, tr = res["equity"], res["trades"]
    if eq.empty:
        return {}
    r = eq.pct_change().fillna(eq.iloc[0] / p.capital - 1)
    half = len(eq) // 2
    out = {
        "期末权益": round(float(eq.iloc[-1])),
        "收益率": round(float(eq.iloc[-1] / p.capital - 1), 3),
        "最大回撤": round(float((eq / eq.cummax() - 1).min()), 3),
        "最差一天": round(float(r.min()), 3),
        "笔数": len(tr),
        "胜率": round(float((tr.pnl > 0).mean()), 2) if len(tr) else 0,
        "前半段": round(float(eq.iloc[half - 1] / p.capital - 1), 3),
        "后半段": round(float(eq.iloc[-1] / eq.iloc[half - 1] - 1), 3),
    }
    return out
