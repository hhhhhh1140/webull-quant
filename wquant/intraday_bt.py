"""日内回测：SOXL / SOXS 做 T。

约定：
- 只用常规时段（RTH）的 5 分钟线；做多用 SOXL，做空用买入 SOXS（系统只做多，不卖空）
- 信号在第 i 根 K 线收盘时产生，第 i+1 根开盘成交（不偷看未来）
- 止损按持有标的的百分比，在每根 K 线内用 high/low 判断；跳空越过止损按开盘价成交
- 每笔双边都计滑点；当天收盘前全部平仓，不留隔夜
- 当天亏损达到上限后停止交易
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Params:
    notional: float = 10_000          # 每笔金额（本金 10 万的 10%）
    capital: float = 100_000
    slippage: float = 0.0005          # 单边滑点 0.05%
    stop_pct: float = 0.03            # 持有标的跌 3% 止损
    daily_loss_limit: float = 0.015   # 当天亏损达到本金 1.5% 停止交易
    max_trades: int = 4
    first_entry: str = "10:00"        # 开盘半小时内不开仓
    last_entry: str = "15:30"


@dataclass
class Trade:
    day: object
    side: int                # +1 = 买 SOXL，-1 = 买 SOXS
    entry_t: pd.Timestamp
    entry: float
    exit_t: pd.Timestamp | None = None
    exit: float | None = None
    reason: str = ""
    qty: float = 0.0

    @property
    def pnl(self) -> float:
        return 0.0 if self.exit is None else (self.exit - self.entry) * self.qty


def load_days(long_df: pd.DataFrame, short_df: pd.DataFrame, ref_df: pd.DataFrame | None = None) -> list[dict]:
    """按日切分常规时段数据，三个标的按时间对齐。"""
    out = []
    l = long_df[long_df.session == "RTH"]
    s = short_df[short_df.session == "RTH"]
    r = ref_df[ref_df.session == "RTH"] if ref_df is not None else None
    for day, ld in l.groupby(l.index.date):
        sd = s[s.index.date == day]
        idx = ld.index.intersection(sd.index)
        if r is not None:
            idx = idx.intersection(r[r.index.date == day].index)
        if len(idx) < 60:          # 缺数据或半天交易日
            continue
        d = {"day": day, "L": ld.loc[idx].copy(), "S": sd.loc[idx].copy()}
        if r is not None:
            d["R"] = r.loc[idx].copy()
        for k in ("L", "S", "R"):
            if k in d:
                f = d[k]
                tp = (f.high + f.low + f.close) / 3
                f["vwap"] = (tp * f.volume).cumsum() / f.volume.cumsum().replace(0, np.nan)
                f["ema9"] = f.close.ewm(span=9, adjust=False).mean()
                f["ema21"] = f.close.ewm(span=21, adjust=False).mean()
        out.append(d)
    return out


# ---------------------------------------------------------------- 策略：返回每根 K 线收盘时的目标仓位 (+1/-1/0)
# 签名：strategy(day, i, current_side, state) -> 目标方向；state 是当天的 dict，可存东西

def orb(day, i, side, st, bars: int = 6):
    """开盘区间突破：前 30 分钟高低点，向上突破做多 SOXL，向下突破买 SOXS；每个方向一天最多一次。"""
    L = day["L"]
    if i < bars:
        return 0
    if "hi" not in st:
        st["hi"], st["lo"], st["used"] = L.high.iloc[:bars].max(), L.low.iloc[:bars].min(), set()
    c = L.close.iloc[i]
    if side != 0:
        return side
    if c > st["hi"] and 1 not in st["used"]:
        st["used"].add(1)
        return 1
    if c < st["lo"] and -1 not in st["used"]:
        st["used"].add(-1)
        return -1
    return 0


def vwap_revert(day, i, side, st, k: float = 0.02, use_filter: bool = False):
    """VWAP 回归：SOXL 低于 VWAP k 以上买 SOXL，高于 VWAP k 以上买 SOXS；回到 VWAP 离场。
    use_filter：只在 QQQ 同方向不弱的时候做（QQQ 在自己 VWAP 上方才抄 SOXL，反之亦然），避免逆着单边行情接刀。"""
    L = day["L"]
    c, v = L.close.iloc[i], L.vwap.iloc[i]
    if side == 1:
        return 0 if c >= v else 1
    if side == -1:
        return 0 if c <= v else -1
    ok_long = ok_short = True
    if use_filter and "R" in day:
        R = day["R"]
        ok_long = R.close.iloc[i] >= R.vwap.iloc[i]
        ok_short = R.close.iloc[i] <= R.vwap.iloc[i]
    if c < v * (1 - k) and ok_long:
        return 1
    if c > v * (1 + k) and ok_short:
        return -1
    return 0


def ema_trend(day, i, side, st):
    """趋势跟随：EMA9 > EMA21 且价格在 VWAP 上方做多 SOXL；反之买 SOXS；条件不满足就平仓。"""
    L = day["L"]
    up = L.ema9.iloc[i] > L.ema21.iloc[i] and L.close.iloc[i] > L.vwap.iloc[i]
    dn = L.ema9.iloc[i] < L.ema21.iloc[i] and L.close.iloc[i] < L.vwap.iloc[i]
    return 1 if up else -1 if dn else 0


STRATEGIES: dict[str, Callable] = {
    "开盘区间突破 ORB": orb,
    "VWAP 回归": lambda d, i, s, st: vwap_revert(d, i, s, st, use_filter=False),
    "VWAP 回归 + QQQ 过滤": lambda d, i, s, st: vwap_revert(d, i, s, st, use_filter=True),
    "EMA 趋势跟随": ema_trend,
}


# ---------------------------------------------------------------- 引擎
def run(days: list[dict], strategy: Callable, p: Params) -> tuple[list[Trade], pd.Series]:
    trades: list[Trade] = []
    daily = {}
    for day in days:
        L, S = day["L"], day["S"]
        times = L.index
        st: dict = {}
        side, pos = 0, None
        day_pnl, n = 0.0, 0
        halted = False
        first = pd.Timestamp(f"{day['day']} {p.first_entry}").tz_localize(times.tz)
        last = pd.Timestamp(f"{day['day']} {p.last_entry}").tz_localize(times.tz)

        def close_pos(j, price, reason):
            nonlocal pos, side, day_pnl
            pos.exit_t, pos.exit, pos.reason = times[j], price * (1 - p.slippage), reason
            day_pnl += pos.pnl
            trades.append(pos)
            pos, side = None, 0

        for i in range(len(times) - 1):
            # 1) 持仓的止损检查（第 i 根 K 线内）
            if pos is not None:
                bar = (L if pos.side == 1 else S).iloc[i]
                stop = pos.entry * (1 - p.stop_pct)
                if bar.low <= stop:
                    close_pos(i, min(bar.open, stop), "止损")
                    if day_pnl <= -p.daily_loss_limit * p.capital:
                        halted = True
            # 2) 收盘产生信号，下一根开盘执行
            want = strategy(day, i, side, st)
            t = times[i]
            if halted or t < first:
                want = side if pos is not None else 0
            if want != side:
                if pos is not None:
                    px = (L if pos.side == 1 else S).open.iloc[i + 1]
                    close_pos(i + 1, px, "信号")
                    if day_pnl <= -p.daily_loss_limit * p.capital:
                        halted = True
                if want != 0 and not halted and n < p.max_trades and t <= last:
                    px = (L if want == 1 else S).open.iloc[i + 1] * (1 + p.slippage)
                    pos = Trade(day["day"], want, times[i + 1], px, qty=p.notional / px)
                    side, n = want, n + 1
        if pos is not None:                          # 收盘平仓
            px = (L if pos.side == 1 else S).close.iloc[-1]
            close_pos(len(times) - 1, px, "收盘平仓")
        daily[day["day"]] = day_pnl
    return trades, pd.Series(daily).sort_index()


def summarize(trades: list[Trade], daily: pd.Series, p: Params) -> dict:
    pnl = np.array([t.pnl for t in trades]) if trades else np.array([0.0])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    eq = p.capital + daily.cumsum()
    dd = (eq / eq.cummax() - 1).min() if len(eq) else 0
    half = len(daily) // 2
    return {
        "总盈亏": round(float(daily.sum()), 0),
        "收益率": round(float(daily.sum() / p.capital), 4),
        "笔数": len(trades),
        "胜率": round(float((pnl > 0).mean()), 3) if trades else 0,
        "平均盈利": round(float(wins.mean()), 0) if len(wins) else 0,
        "平均亏损": round(float(losses.mean()), 0) if len(losses) else 0,
        "盈亏比(PF)": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
        "最大回撤": round(float(dd), 4),
        "日夏普": round(float(daily.mean() / daily.std() * np.sqrt(252)), 2) if daily.std() > 0 else 0,
        "前半段": round(float(daily.iloc[:half].sum()), 0),
        "后半段": round(float(daily.iloc[half:].sum()), 0),
        "最好一天": round(float(daily.max()), 0),
        "最差一天": round(float(daily.min()), 0),
    }


def buy_hold_intraday(days: list[dict], key: str, p: Params) -> pd.Series:
    """基准：每天开盘买、收盘卖（同样金额和滑点）。"""
    out = {}
    for d in days:
        f = d[key]
        o, c = f.open.iloc[0] * (1 + p.slippage), f.close.iloc[-1] * (1 - p.slippage)
        out[d["day"]] = (c - o) * p.notional / o
    return pd.Series(out).sort_index()
