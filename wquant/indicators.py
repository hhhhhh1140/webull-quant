"""技术指标（纯 pandas，无未来函数：t 时刻只用 t 及之前的数据）。"""
from __future__ import annotations

import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(loss.notna())


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def add_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    out = df.copy()
    out["sma_fast"] = sma(out["close"], p.get("sma_fast", 50))
    out["sma_slow"] = sma(out["close"], p.get("sma_slow", 200))
    out["rsi"] = rsi(out["close"], p.get("rsi_period", 14))
    out["atr"] = atr(out, p.get("atr_period", 14))
    lb = p.get("momentum_lookback", 20)
    out["momentum"] = out["close"] / out["close"].shift(lb) - 1
    return out
