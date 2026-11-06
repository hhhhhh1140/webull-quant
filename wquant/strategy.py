"""规则层：趋势 + 动量筛选，产生买入候选和卖出（止损/离场）信号。

AI 只在这里给出的候选里做取舍，不能凭空选股；止损信号默认不可被 AI 否决。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Signal:
    symbol: str
    action: str          # BUY | SELL
    reason: str
    price: float
    atr: float


def entry_ok(row: pd.Series, p: dict) -> bool:
    if row[["sma_fast", "sma_slow", "rsi", "atr", "momentum"]].isna().any():
        return False
    return bool(
        row["close"] > row["sma_fast"] > row["sma_slow"]
        and row["rsi"] < p.get("rsi_max_entry", 70)
        and row["momentum"] > 0
    )


def exit_reason(row: pd.Series, p: dict, trail_high: float) -> str | None:
    """trail_high：持仓以来的最高收盘价，用于 ATR 跟踪止损。"""
    stop = trail_high - p.get("atr_stop_mult", 2.5) * row["atr"]
    if row["close"] < stop:
        return f"ATR 跟踪止损 (close {row['close']:.2f} < stop {stop:.2f})"
    if p.get("exit_below_fast_sma", True) and row["close"] < row["sma_fast"]:
        return f"跌破 SMA{p.get('sma_fast', 50)}"
    return None


def market_regime(frames: dict[str, pd.DataFrame], p: dict, at: pd.Timestamp | None = None) -> dict:
    """大盘趋势过滤：基准指数跌破长期均线时视为弱势，停止开新仓（卖出/止损不受影响）。

    frames 需已含指标；at 为 None 时取每个 DataFrame 的最后一行（实盘），否则取该日（回测）。
    """
    cfg = p.get("market_filter") or {}
    if not cfg.get("enabled", False):
        return {"enabled": False, "risk_off": False, "indices": []}
    ma = cfg.get("ma", "sma_slow")            # sma_slow = SMA200，sma_fast = SMA50
    indices, weak = [], []
    for sym in cfg.get("symbols", ["SPY", "QQQ"]):
        df = frames.get(sym)
        if df is None or df.empty:
            continue
        if at is None:
            row = df.iloc[-1]
        elif at in df.index:
            row = df.loc[at]
        else:
            continue
        if pd.isna(row[ma]):
            continue
        below = bool(row["close"] < row[ma])
        indices.append({"symbol": sym, "close": round(float(row["close"]), 2), "ma": round(float(row[ma]), 2),
                        "ma_name": "SMA" + str(p.get("sma_slow" if ma == "sma_slow" else "sma_fast")),
                        "pct_vs_ma": round(float(row["close"] / row[ma] - 1), 4), "below": below})
        if below:
            weak.append(sym)
    mode = cfg.get("mode", "any")             # any = 任一指数走弱就停；all = 全部走弱才停
    risk_off = bool(weak) and (mode == "any" or len(weak) == len(indices))
    return {"enabled": True, "risk_off": risk_off, "mode": mode, "weak": weak, "indices": indices}


# ---------------------------------------------------------------- 反向 ETF 短线规则
# 3 倍反向 ETF 长期损耗，价格大部分时间在 200 日线下方，普通趋势规则几乎永远不会选中它。
# 所以单独一套：只在大盘走弱时用 20/50 日均线做短线，持有时间有上限，止损更紧。

def inverse_entry_ok(row: pd.Series, p: dict, index_weak: bool) -> bool:
    r = p.get("inverse_rule") or {}
    if not r.get("enabled", True):
        return False
    if row[["st_fast", "st_slow", "st_mom", "rsi", "atr"]].isna().any():
        return False
    if r.get("require_weak_index", True) and not index_weak:
        return False
    return bool(
        row["close"] > row["st_fast"] > row["st_slow"]
        and row["st_mom"] > 0
        and row["rsi"] < r.get("rsi_max_entry", 80)
    )


def inverse_exit_reason(row: pd.Series, p: dict, trail_high: float, hold_days: int | None) -> str | None:
    r = p.get("inverse_rule") or {}
    stop = trail_high - r.get("atr_stop_mult", 2.0) * row["atr"]
    if row["close"] < stop:
        return f"ATR 跟踪止损 (close {row['close']:.2f} < stop {stop:.2f})"
    if row["close"] < row["st_fast"]:
        return f"跌破 SMA{r.get('sma_fast', 20)}"
    max_hold = r.get("max_hold_days", 15)
    if hold_days is not None and hold_days >= max_hold:
        return f"持有满 {hold_days} 个交易日（反向 ETF 持有上限 {max_hold} 天）"
    return None


def hedge_exit_reason(row: pd.Series, cfg: dict, trail_high: float, hold_days: int | None,
                      view_regime: str | None) -> str | None:
    """对冲仓位的退出：ATR 止损 / 持有天数上限 / 市场研判不再偏空。"""
    stop = trail_high - cfg.get("atr_stop_mult", 2.0) * row["atr"]
    if row["close"] < stop:
        return f"对冲 ATR 止损 (close {row['close']:.2f} < stop {stop:.2f})"
    max_hold = cfg.get("max_hold_days", 10)
    if hold_days is not None and hold_days >= max_hold:
        return f"对冲持有满 {hold_days} 个交易日（上限 {max_hold}）"
    if cfg.get("exit_when_view_changes", True) and view_regime is not None and view_regime != "risk_off":
        return f"市场研判已转为 {view_regime}，平掉对冲"
    return None


def index_weak(frames: dict[str, pd.DataFrame], p: dict, at: pd.Timestamp | None = None) -> bool:
    """短线视角的大盘走弱：任一基准指数跌破 SMA50。供反向 ETF 入场使用。"""
    mf = p.get("market_filter") or {}
    q = {**p, "market_filter": {"enabled": True, "symbols": mf.get("symbols", ["SPY", "QQQ"]),
                                "ma": "sma_fast", "mode": "any"}}
    return market_regime(frames, q, at)["risk_off"]


def hold_days_between(entry: str | pd.Timestamp | None, today: pd.Timestamp) -> int | None:
    if entry is None:
        return None
    import numpy as np
    return int(np.busday_count(pd.Timestamp(entry).date(), pd.Timestamp(today).date()))


def generate_signals(frames: dict[str, pd.DataFrame], held: dict[str, float], p: dict,
                     inverse: set[str] | None = None, entry_dates: dict[str, str] | None = None,
                     weak_index: bool = False) -> list[Signal]:
    """frames 已含指标；held = {symbol: 持仓以来最高收盘价}。只看每个 DataFrame 的最后一行。
    inverse：反向 ETF 代码集合，走短线规则；entry_dates：{symbol: 买入日期}，用于持有天数上限。"""
    inverse = inverse or set()
    entry_dates = entry_dates or {}
    sigs: list[Signal] = []
    for sym, df in frames.items():
        if df.empty:
            continue
        row = df.iloc[-1]
        inv = sym in inverse
        if sym in held:
            trail = max(held[sym], row["close"])
            if inv:
                r = inverse_exit_reason(row, p, trail, hold_days_between(entry_dates.get(sym), df.index[-1]))
            else:
                r = exit_reason(row, p, trail)
            if r:
                sigs.append(Signal(sym, "SELL", r, float(row["close"]), float(row["atr"])))
        elif inv:
            if inverse_entry_ok(row, p, weak_index):
                rr = p.get("inverse_rule") or {}
                sigs.append(Signal(sym, "BUY", f"反向 ETF 短线：大盘走弱 + 站上 SMA{rr.get('sma_fast', 20)}/"
                                   f"{rr.get('sma_slow', 50)} + 动量为正", float(row["close"]), float(row["atr"])))
        elif entry_ok(row, p):
            sigs.append(Signal(sym, "BUY", "趋势向上 + 动量为正 + RSI 未超买", float(row["close"]), float(row["atr"])))
    return sigs
