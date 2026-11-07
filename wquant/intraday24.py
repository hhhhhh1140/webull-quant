"""跨时段（夜盘 / 盘前 / 常规）回测：夜盘方向 → 顺势持有。

每个交易日 D 的窗口：D-1 常规时段收盘（16:00 ET）→ D 常规时段收盘。
流程：
1. 在入场时间 entry（如 04:00 ET，夜盘结束）看 SOXL 相对 D-1 收盘的涨跌：
   ≥ +threshold 买 SOXL；≤ −threshold 买 SOXS；可选要求韩国（EWY）同方向确认
2. 下一根 K 线开盘成交；持有到 exit（09:30 开盘 或 16:00 收盘）
3. 止损：固定 stop_pct 或从持仓最高点回撤 trail_pct（按持有标的价格）
4. 止损后可以再买回（reentry）：被止损后标的继续跌，从止损后的最低点反弹 rebound_pct 就按原方向买回一次
5. 滑点按时段：常规时段 slip_rth，其他时段 slip_ext
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass
class P24:
    entry: str = "04:00"            # 入场判断时间（美东）
    exit: str = "16:00"             # "09:30" = 开盘卖；"16:00" = 收盘卖
    threshold: float = 0.03         # SOXL 相对昨收涨跌超过这个才做
    korea_confirm: float | None = None   # EWY 同方向至少这么多才做；None = 不看韩国
    stop_pct: float | None = 0.06   # 固定止损（持有标的跌多少）
    trail_pct: float | None = None  # 移动止损（从持仓最高点回撤多少）
    reentry: bool = False
    rebound_pct: float = 0.03       # 止损后从最低点反弹多少再买回
    last_reentry: str = "15:00"
    notional: float = 10_000
    capital: float = 100_000
    slip_rth: float = 0.0005
    slip_ext: float = 0.0015


def _slip(sess: str, p: P24) -> float:
    return p.slip_rth if sess == "RTH" else p.slip_ext


def _at(df: pd.DataFrame, t: pd.Timestamp) -> int | None:
    """第一根时间 ≥ t 的 K 线位置。"""
    i = df.index.searchsorted(t)
    return int(i) if i < len(df) else None


def build_days(L: pd.DataFrame, S: pd.DataFrame, K: pd.DataFrame | None = None) -> list[dict]:
    rth = L[L.session == "RTH"]
    days = sorted(set(rth.index.date))
    out = []
    for prev, d in zip(days, days[1:]):
        r_prev, r = rth[rth.index.date == prev], rth[rth.index.date == d]
        if len(r_prev) < 60 or len(r) < 60:
            continue
        t0, t1 = r_prev.index[-1], r.index[-1]
        item = {"day": d, "t_prev_close": t0, "t_open": r.index[0], "t_close": t1,
                "L": L[(L.index > t0) & (L.index <= t1)], "S": S[(S.index > t0) & (S.index <= t1)],
                "L_prev_close": r_prev.close.iloc[-1]}
        if K is not None:
            k_rth = K[K.session == "RTH"]
            kp = k_rth[k_rth.index.date == prev]
            if len(kp):
                item["K"] = K[(K.index > t0) & (K.index <= t1)]
                item["K_prev_close"] = kp.close.iloc[-1]
        if len(item["L"]) > 100 and len(item["S"]) > 100:
            out.append(item)
    return out


def _ts(day, hhmm: str, tz) -> pd.Timestamp:
    h, m = map(int, hhmm.split(":"))
    base = pd.Timestamp(day).tz_localize(tz)
    t = base + pd.Timedelta(hours=h, minutes=m)
    if h >= 16:                        # 16:00 以后的时间属于前一天晚上（夜盘从 20:00 开始）
        t -= pd.Timedelta(days=1)
    return t


def run24(days: list[dict], p: P24) -> tuple[list[dict], pd.Series]:
    trades, daily = [], {}
    for d in days:
        L, S = d["L"], d["S"]
        tz = L.index.tz
        t_entry = _ts(d["day"], p.entry, tz)
        t_exit = d["t_open"] if p.exit == "09:30" else d["t_close"]
        i = _at(L, t_entry)
        day_pnl = 0.0
        if i is None or L.index[i] >= t_exit:
            daily[d["day"]] = 0.0
            continue
        move = L.close.iloc[i] / d["L_prev_close"] - 1
        side = 1 if move >= p.threshold else -1 if move <= -p.threshold else 0
        if side and p.korea_confirm is not None:
            K = d.get("K")
            ki = _at(K, t_entry) if K is not None else None
            if ki is None:
                side = 0
            else:
                kmove = K.close.iloc[ki] / d["K_prev_close"] - 1
                if np.sign(kmove) != side or abs(kmove) < p.korea_confirm:
                    side = 0
        if side == 0:
            daily[d["day"]] = 0.0
            continue
        X = L if side == 1 else S
        j = _at(X, L.index[i] + pd.Timedelta(minutes=5))
        reentries = 0
        while j is not None and X.index[j] < t_exit:
            # ---- 开仓
            entry_px = X.open.iloc[j] * (1 + _slip(X.session.iloc[j], p))
            qty = p.notional / entry_px
            peak = entry_px
            exit_px, reason, k = None, "", j
            for k in range(j, len(X)):
                if X.index[k] >= t_exit:
                    break
                bar = X.iloc[k]
                stops = []
                if p.stop_pct:
                    stops.append(entry_px * (1 - p.stop_pct))
                if p.trail_pct:
                    stops.append(peak * (1 - p.trail_pct))
                stop = max(stops) if stops else None
                if stop is not None and bar.low <= stop:
                    exit_px, reason = min(bar.open, stop) * (1 - _slip(bar.session, p)), "止损"
                    break
                peak = max(peak, bar.high)
            if exit_px is None:          # 到时间平仓：用退出时间之前最后一根的收盘
                last = X[X.index < t_exit] if p.exit == "09:30" else X[X.index <= t_exit]
                exit_px = last.close.iloc[-1] * (1 - _slip(last.session.iloc[-1], p))
                reason = "到时平仓"
            pnl = (exit_px - entry_px) * qty
            day_pnl += pnl
            trades.append({"day": d["day"], "side": side, "entry_t": X.index[j], "entry": entry_px,
                           "exit": exit_px, "pnl": pnl, "reason": reason, "reentry": reentries})
            if reason != "止损" or not p.reentry or reentries >= 1:
                break
            # ---- 止损后：等最低点反弹 rebound_pct 再买回
            low, j = X.low.iloc[k], None
            cutoff = _ts(d["day"], p.last_reentry, tz)
            for m in range(k + 1, len(X)):
                if X.index[m] >= min(t_exit, cutoff):
                    break
                low = min(low, X.low.iloc[m])
                if X.close.iloc[m] >= low * (1 + p.rebound_pct):
                    j = m + 1 if m + 1 < len(X) else None
                    break
            reentries += 1
        daily[d["day"]] = day_pnl
    return trades, pd.Series(daily).sort_index()


def stats(trades: list[dict], daily: pd.Series, p: P24) -> dict:
    pnl = np.array([t["pnl"] for t in trades]) if trades else np.zeros(1)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    eq = p.capital + daily.cumsum()
    half = len(daily) // 2
    big = daily.abs().nlargest(3).index if len(daily) else []
    return {
        "总盈亏": round(float(daily.sum())), "笔数": len(trades),
        "胜率": round(float((pnl > 0).mean()), 2) if trades else 0,
        "PF": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
        "最大回撤": round(float((eq / eq.cummax() - 1).min()), 3),
        "前半段": round(float(daily.iloc[:half].sum())), "后半段": round(float(daily.iloc[half:].sum())),
        "去掉最大3天": round(float(daily.drop(big).sum())),
        "中位数/笔": round(float(np.median(pnl)), 0),
    }
