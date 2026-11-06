"""数据层：从 Webull 拉日线并缓存到本地，也支持 CSV 导入。"""
from __future__ import annotations

from pathlib import Path

import zlib

import numpy as np
import pandas as pd

from .config import ROOT, Settings

CACHE = ROOT / "data" / "cache"


def _cache_path(symbol: str) -> Path:
    return CACHE / f"{symbol.upper()}.csv"


def save_cache(symbol: str, df: pd.DataFrame) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(_cache_path(symbol), index_label="date")


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()


def load_bars(settings: Settings, symbols: list[str], source: str = "webull", count: int = 1200,
              refresh: bool = True) -> dict[str, pd.DataFrame]:
    """source: webull | cache | synthetic"""
    if source == "synthetic":
        return {s: synthetic_bars(s, n=count) for s in symbols}
    if source == "cache":
        out = {}
        for s in symbols:
            p = _cache_path(s)
            if p.exists():
                out[s] = load_csv(p)
        missing = set(symbols) - set(out)
        if missing:
            print(f"[data] 缓存中缺少: {sorted(missing)}（先运行 `wq fetch` 拉取）")
        return out
    from .webull_client import WebullClient

    client = WebullClient(settings)
    bars = client.daily_bars(symbols, count=count)
    if refresh:
        for s, df in bars.items():
            save_cache(s, df)
    return bars


def synthetic_bars(symbol: str, n: int = 1200, seed: int | None = None) -> pd.DataFrame:
    """几何布朗运动生成的假数据——只用于测试和离线演示，不代表任何真实行情。"""
    rng = np.random.default_rng(seed if seed is not None else zlib.crc32(symbol.encode()))
    drift, vol = rng.uniform(0.0, 0.0006), rng.uniform(0.01, 0.025)
    rets = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, vol / 4, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": rng.integers(1e6, 5e6, n)},
        index=idx,
    ).astype(float)
