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
            # ---- 检查时点：入场 / 反向
            if t in cps and not halted:
                mv = win.close.loc[t] / prev_close - 1
                sig = 1 if mv >= p.threshold else -1 if mv <= -p.threshold else 0
                kmv = None
                if sig and p.korea_confirm is not None and K is not None and ("prev_close" in K or kprev is not None):
                    if "prev_close" in K:            # KOSPI 指数：自带上一交易日收盘
                        from .intraday_data import kospi_move
                        kmv = kospi_move(K, t) or 0.0
                    else:                            # EWY 等美股 ETF：相对其美股昨收
                        ki = K.index.searchsorted(t, side="right") - 1
                        kmv = K.close.iloc[ki] / kprev - 1 if ki >= 0 else 0
                    if np.sign(kmv) != sig or abs(kmv) < p.korea_confirm:
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
        # ---- 当天收盘：不持仓过夜，或偏向不一致 → 平仓
        if pos is not None:
            X = inst[pos["side"]]
            keep = p.carry and b == pos["side"]
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
