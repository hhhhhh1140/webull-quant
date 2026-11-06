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


def generate_signals(frames: dict[str, pd.DataFrame], held: dict[str, float], p: dict) -> list[Signal]:
    """frames 已含指标；held = {symbol: 持仓以来最高收盘价}。只看每个 DataFrame 的最后一行。"""
    sigs: list[Signal] = []
    for sym, df in frames.items():
        if df.empty:
            continue
        row = df.iloc[-1]
        if sym in held:
            r = exit_reason(row, p, max(held[sym], row["close"]))
            if r:
                sigs.append(Signal(sym, "SELL", r, float(row["close"]), float(row["atr"])))
        elif entry_ok(row, p):
            sigs.append(Signal(sym, "BUY", "趋势向上 + 动量为正 + RSI 未超买", float(row["close"]), float(row["atr"])))
    return sigs
