"""固定规则的日内入场信号，输入为从 09:30 起对齐的 SOXL / SOXS 常规时段 M5 线。"""
from __future__ import annotations

import numpy as np
import pandas as pd


RTH_RULES = ("opening_range", "trend_pullback", "vwap_reclaim")


def _vwap_reclaim(frame):
    typical = (frame.high + frame.low + frame.close) / 3
    vwap = (typical * frame.volume).cumsum() / frame.volume.cumsum().replace(0, np.nan)
    ratio = frame.close / vwap
    # 收回折价带时仍留有回到 VWAP 的空间，同时要求标的自身价格已经回升。
    signal = ((ratio.shift() <= 0.99) & (ratio > 0.99) & (ratio < 1)
              & (frame.close > frame.close.shift()))
    signal.iloc[:6] = False
    upside = (vwap / frame.close - 1).where(signal, 0.0)
    return upside.to_numpy(), vwap.to_numpy()


def signals(name: str, L: pd.DataFrame, S: pd.DataFrame, trend=None) -> tuple[np.ndarray, np.ndarray]:
    """返回收盘时的入场方向和标的原价目标。方向为 +1 SOXL、-1 SOXS、0 观望。

    下一根开盘才可入场。目标为 NaN 时使用撮合引擎的常规止盈；VWAP 目标固定在信号时。
    trend 是调用方提供的前一日趋势，这三种纯日内规则均仅使用本交易时段的数据。
    """
    if name not in RTH_RULES:
        raise ValueError(f"未知日内策略 {name}")
    if not L.index.equals(S.index):
        raise ValueError("SOXL 和 SOXS 的 K 线时间不一致")
    sides = np.zeros(len(L), dtype=np.int8)
    targets = np.full(len(L), np.nan)
    if len(L) <= 6:
        return sides, targets

    if name == "opening_range":
        # 前六根定区间，之后以首个超出区间的收盘确认，每个方向每天只发一次信号。
        later = L.close.to_numpy()[6:]
        up = np.flatnonzero(later > L.high.iloc[:6].max())[:1]
        down = np.flatnonzero(later < L.low.iloc[:6].min())[:1]
        sides[6 + up] = 1
        sides[6 + down] = -1
    elif name == "trend_pullback":
        close = L.close
        fast = close.ewm(span=9, adjust=False, min_periods=9).mean()
        slow = close.ewm(span=21, adjust=False, min_periods=21).mean()
        previous = close.shift()
        up = (fast > slow) & (previous <= fast.shift()) & (close > fast) & (close > L.high.shift())
        down = (fast < slow) & (previous >= fast.shift()) & (close < fast) & (close < L.low.shift())
        sides[up] = 1
        sides[down] = -1
    else:
        long_upside, long_vwap = _vwap_reclaim(L)
        short_upside, short_vwap = _vwap_reclaim(S)
        buy_long = long_upside > short_upside
        buy_short = short_upside > long_upside
        sides[buy_long] = 1
        sides[buy_short] = -1
        targets[buy_long] = long_vwap[buy_long]
        targets[buy_short] = short_vwap[buy_short]
    return sides, targets
