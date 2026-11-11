"""分钟级数据：从 Webull 往回翻页下载（最多约 180 天），存到 data/intraday/，之后增量更新。"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .config import ROOT

DIR = ROOT / "data" / "intraday"
SESSIONS = "PRE,RTH,ATH,OVN"


def path(symbol: str, timespan: str) -> Path:
    return DIR / f"{symbol.upper()}_{timespan}.csv"


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/New_York")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df = df.rename(columns={"trading_session": "session"})
    return df.set_index("t")[["open", "high", "low", "close", "volume", "session"]].sort_index()


def load(symbol: str, timespan: str = "M5", adjust: bool = True) -> pd.DataFrame:
    p = path(symbol, timespan)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    df = df.sort_index()
    return adjust_splits(df) if adjust else df


_COMMON = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100)


def _snap(r: float) -> float:
    """把跳变比例对齐到最近的常见拆合股比例（比例本身混了当天的涨跌，不能直接四舍五入）。"""
    import math
    x = r if r > 1 else 1 / r
    best = min(_COMMON, key=lambda k: abs(math.log(x / k)))
    return best if r > 1 else 1 / best


def adjust_splits(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    """Webull 分钟线没有复权。相邻两根 K 线价格跳变超过 threshold 倍，视为拆股 / 合股，
    把之前的价格按整数比例调整（成交量反向调整），让跨时段持仓的收益计算正确。"""
    df = df.copy()
    ratio = df["open"] / df["close"].shift()
    for t, r in ratio[(ratio > threshold) | (ratio < 1 / threshold)].items():
        factor = _snap(r)
        before = df.index < t
        for c in ("open", "high", "low", "close"):
            df.loc[before, c] = df.loc[before, c] * factor
        df.loc[before, "volume"] = df.loc[before, "volume"] / factor
    return df


def download(client, symbol: str, timespan: str = "M5", days: int = 178, pause: float = 0.3) -> pd.DataFrame:
    """从现在往回翻页，直到 days 天前（接口限制 180 天）。与已有文件合并。"""
    from .webull_client import _json, _unwrap_list
    stop = datetime.now(timezone.utc) - timedelta(days=days)
    end = None
    parts = []
    while True:
        kw = {"count": "1200", "trading_sessions": SESSIONS}
        if end is not None:
            kw["end_time"] = int(end.timestamp() * 1000)
        data = _unwrap_list(_json(client.data.market_data.get_batch_history_bar([symbol], "US_STOCK", timespan, **kw)))
        rows = data[0]["result"] if data else []
        df = _to_frame(rows)
        if df.empty:
            break
        parts.append(df)
        first = df.index.min().tz_convert("UTC").to_pydatetime()
        if first <= stop or (end is not None and first >= end):
            break
        end = first - timedelta(seconds=1)
        time.sleep(pause)
    new = pd.concat(parts) if parts else pd.DataFrame()
    old = load(symbol, timespan)
    all_ = pd.concat([old, new]) if not old.empty else new
    all_ = all_[~all_.index.duplicated(keep="last")].sort_index()
    DIR.mkdir(parents=True, exist_ok=True)
    all_.to_csv(path(symbol, timespan))
    return all_


# ---------------------------------------------------------------- 韩国综合指数 KOSPI（Webull 不提供，用 Yahoo Finance 公开行情）
KOSPI_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EKS11"
_RANGE = {"5m": ("5m", "60d", 5), "1h": ("60m", "730d", 60), "1d": ("1d", "5y", None)}


def fetch_kospi(interval: str = "1h") -> pd.DataFrame:
    """拉取 KOSPI，索引改成"这根 K 线收盘的时间"（美东），避免回测偷看未来。
    prev_close 列 = 上一个韩国交易日的收盘，用来算当天涨跌。与已有缓存合并。"""
    import requests
    yi, rng, minutes = _RANGE[interval]
    r = requests.get(KOSPI_URL, params={"interval": yi, "range": rng},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    j = r.json()["chart"]["result"][0]
    ts = pd.to_datetime(j["timestamp"], unit="s", utc=True)
    q = j["indicators"]["quote"][0]
    df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]}, index=ts).dropna()
    if minutes:
        df.index = df.index + pd.Timedelta(minutes=minutes)       # K 线收盘才算已知
    p = path("KOSPI", interval)
    if p.exists():
        old = pd.read_csv(p, index_col=0)
        old.index = pd.to_datetime(old.index, utc=True)
        df = pd.concat([old[["open", "high", "low", "close"]], df])
        df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(p)
    return load_kospi(interval)


def load_kospi(interval: str = "1h") -> pd.DataFrame:
    p = path("KOSPI", interval)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    kst = df.index.tz_convert("Asia/Seoul")
    # 收盘时间 15:30 那根 K 线的"收盘时刻"可能落在 15:30 之后，按开盘时间所在的日期归属交易日
    minutes = {"5m": 5, "1h": 60}.get(interval, 0)
    kday = (kst - pd.Timedelta(minutes=minutes)).date
    df["kday"] = kday
    day_close = df.groupby("kday").close.last()
    if interval != "1d":
        # 用日线的官方收盘价（含收盘集合竞价）；分钟线最后一根可能和官方收盘差很多
        official = load_kospi("1d")
        if not official.empty:
            oc = official.groupby("kday").close.last()
            day_close = oc.reindex(day_close.index).fillna(day_close)
    prev = day_close.shift(1)
    df["prev_close"] = df["kday"].map(prev)
    df.index = df.index.tz_convert("America/New_York")
    return df.dropna(subset=["prev_close"])


def kospi_move(K: pd.DataFrame, t) -> float | None:
    """时刻 t（美东）已知的 KOSPI 当天涨跌：最近一根已收盘 K 线相对上一交易日收盘。"""
    i = K.index.searchsorted(t, side="right") - 1
    if i < 0:
        return None
    return float(K.close.iloc[i] / K.prev_close.iloc[i] - 1)


def kospi_path(K: pd.DataFrame, t) -> tuple[float, float, float] | None:
    """时刻 t（美东）已知的 KOSPI 当天走势：(相对昨收, 相对当天最低, 相对当天最高)，只用已收盘的 K 线。"""
    i = K.index.searchsorted(t, side="right") - 1
    if i < 0:
        return None
    kd = K.kday.iloc[i]
    j = i
    while j > 0 and K.kday.iloc[j - 1] == kd:
        j -= 1
    seg = K.iloc[j:i + 1]
    c = float(seg.close.iloc[-1])
    return c / float(seg.prev_close.iloc[-1]) - 1, c / float(seg.low.min()) - 1, c / float(seg.high.max()) - 1
