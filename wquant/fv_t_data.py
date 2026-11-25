"""盘中 5 分钟动态公允价做 T（fv-t）的分钟级数据层。

- 历史回测：从 Webull 拉取常规时段（RTH）5 分钟 K 线（支持约 60~120 天），缓存到 data/intraday/fv_t/
- 实盘/模拟盘轮询：一次批量请求拉取整个做 T 标的池 + SPY/QQQ 当天及前几日最新 M5 K 线
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, Settings
from .intraday_data import _to_frame, adjust_splits
from .scalp_data import ET, M5, session_bounds
from .webull_client import _json, _unwrap_list

log = logging.getLogger(__name__)

FV_T_DIR = ROOT / "data" / "intraday" / "fv_t"
OHLCV = ["open", "high", "low", "close", "volume"]


def fv_t_symbols(settings: Settings, groups_allowed: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
    """根据配置的板块白名单返回 (交易标的列表, {symbol: group})。"""
    allowed = set(groups_allowed or ["semis", "memory", "optical"])
    grp_map = {s: g for s, g in settings.groups.items() if g in allowed}
    return list(grp_map.keys()), grp_map


def cache_path(symbol: str) -> Path:
    return FV_T_DIR / f"{symbol.upper()}_M5.csv"


def load_cached_m5(symbol: str, adjust: bool = True) -> pd.DataFrame:
    p = cache_path(symbol)
    if not p.exists():
        # 兼容复用 data/intraday/<SYM>_M5.csv
        alt = ROOT / "data" / "intraday" / f"{symbol.upper()}_M5.csv"
        if not alt.exists():
            return pd.DataFrame()
        p = alt
    df = pd.read_csv(p, index_col=0)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
    df = df.sort_index()
    return adjust_splits(df) if adjust else df


def download_m5_rth(client, symbol: str, days: int = 90, pause: float = 0.25) -> pd.DataFrame:
    """从 Webull 往回翻页下载单只标的最近 days 天的常规时段（RTH）M5 K 线并缓存。"""
    started = datetime.now(timezone.utc)
    stop = started - timedelta(days=days)
    end = None
    parts: list[pd.DataFrame] = []

    for _ in range(12):  # 90 天 RTH ≈ 65 个交易日 × 78 根 ≈ 5070 根，约 5 页
        kw: dict = {"count": "1200", "trading_sessions": "RTH"}
        if end is not None:
            kw["end_time"] = int(end.timestamp() * 1000)
        data = _unwrap_list(_json(client.data.market_data.get_batch_history_bar([symbol], "US_STOCK", "M5", **kw)))
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
    old = load_cached_m5(symbol, adjust=False)
    merged = pd.concat([old, new]) if not old.empty else new
    if merged.empty:
        return merged
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    # 剔除尚未收盘的最后一根未完结 M5
    last_closed = started - timedelta(minutes=5)
    merged = merged.loc[merged.index <= last_closed]
    FV_T_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path(symbol))
    return adjust_splits(merged)


def fetch_m5_universe(
    settings: Settings,
    symbols: list[str],
    days: int = 90,
    refresh: bool = True,
) -> dict[str, pd.DataFrame]:
    """批量加载（或下载更新）回测所需的 M5 数据（自动包含 SPY/QQQ）。"""
    all_syms = list(dict.fromkeys(["SPY", "QQQ"] + [s.upper() for s in symbols]))
    out: dict[str, pd.DataFrame] = {}
    if not refresh:
        for s in all_syms:
            df = load_cached_m5(s)
            if not df.empty:
                out[s] = df
        return out

    from .webull_client import WebullClient

    client = WebullClient(settings)
    for i, s in enumerate(all_syms, 1):
        try:
            df = download_m5_rth(client, s, days=days)
            if not df.empty:
                out[s] = df
                print(f"  [{i:2d}/{len(all_syms)}] {s:<5} {len(df):5d} 根 M5 ({df.index[0].date()} → {df.index[-1].date()})")
        except Exception as e:
            log.warning("拉取 %s M5 失败，回退缓存: %s", s, e)
            df = load_cached_m5(s)
            if not df.empty:
                out[s] = df
    return out


def fetch_live_m5_batch(client, symbols: list[str], count: int = 900) -> dict[str, pd.DataFrame]:
    """模拟盘实时轮询：单次批量拉取全部标的最近约 11 个交易日（900 根）RTH M5 K 线（用于滚动 Beta + 今日实时信号）。"""
    all_syms = list(dict.fromkeys(["SPY", "QQQ"] + [s.upper() for s in symbols]))
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(all_syms), 20):
        batch = all_syms[i : i + 20]
        raw = _json(
            client.data.market_data.get_batch_history_bar(
                batch,
                "US_STOCK",
                "M5",
                count=str(min(count, 1200)),
                trading_sessions="RTH",
                real_time_required=True,
            )
        )
        for item in _unwrap_list(raw):
            sym = str(item.get("symbol") or "").upper()
            rows = item.get("result") or []
            if sym and rows:
                df = _to_frame(rows)
                if not df.empty:
                    out[sym] = adjust_splits(df)
    return out


def split_rth_by_day(bars_map: dict[str, pd.DataFrame]) -> dict[date, dict[str, pd.DataFrame]]:
    """将多只标的的 M5 数据按美东交易日分组，仅保留常规时段（RTH）且去重排序。"""
    by_day: dict[date, dict[str, pd.DataFrame]] = {}
    for sym, df in bars_map.items():
        if df.empty:
            continue
        rth = df[df["session"] == "RTH"][OHLCV] if "session" in df.columns else df[OHLCV]
        if rth.empty:
            continue
        rth = rth.tz_convert(ET) if rth.index.tz is not None else rth.tz_localize(ET)
        rth = rth[~rth.index.duplicated(keep="last")].sort_index()
        for day, grp in rth.groupby(rth.index.date):
            if len(grp) >= 12:  # 跳过残缺日
                by_day.setdefault(day, {})[sym] = grp
    return dict(sorted(by_day.items()))


def latest_closed_cutoff(now_et: pd.Timestamp) -> pd.Timestamp:
    """返回当前时刻之前最近一根已经完结的 5 分钟 K 线开盘时刻（不含正在跳动的未收盘 K 线）。"""
    return now_et.floor("5min") - M5


def is_rth_open(now_et: pd.Timestamp) -> tuple[bool, pd.Timestamp | None, pd.Timestamp | None]:
    bounds = session_bounds(now_et.date())
    if not bounds:
        return False, None, None
    start, close = bounds
    return bool(start <= now_et <= close), start, close


def synthetic_m5_universe(
    symbols: list[str],
    groups: dict[str, str],
    days: int = 35,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """生成带真实板块联动 + 盘中均值回归噪声的合成 5 分钟 K 线（仅用于单元测试与离线演示）。"""
    rng = np.random.default_rng(seed)
    bdays = pd.bdate_range("2026-08-03", periods=days)
    all_syms = list(dict.fromkeys(["SPY", "QQQ"] + symbols))

    # 构造每个交易日的 78 根 M5 时间戳（09:30 ~ 15:55 ET）
    stamps = []
    for d in bdays:
        s = pd.Timestamp(d.date(), tz=ET) + pd.Timedelta(hours=9, minutes=30)
        stamps.extend(pd.date_range(s, periods=78, freq=M5))
    idx = pd.DatetimeIndex(stamps)
    n = len(idx)

    mkt = rng.normal(0.00002, 0.0012, n)
    grp_factors = {g: rng.normal(0.0, 0.0020, n) for g in set(groups.values())}

    out: dict[str, pd.DataFrame] = {}
    for sym in all_syms:
        g = groups.get(sym, "index")
        if sym == "SPY":
            lp = np.cumsum(mkt)
        elif sym == "QQQ":
            lp = np.cumsum(1.15 * mkt + rng.normal(0, 0.0005, n))
        else:
            # 板块因子 + 均值回归 OU 特异残差（模拟盘中流动性错杀后回归）
            ou = np.zeros(n)
            for k in range(1, n):
                ou[k] = 0.78 * ou[k - 1] + rng.normal(0, 0.0035)
            lp = np.cumsum(0.9 * mkt + 1.0 * grp_factors[g]) + ou
        close = 100.0 * np.exp(lp)
        open_ = np.r_[close[0], close[:-1]]
        spread = np.abs(rng.normal(0.0008, 0.0006, n))
        high = np.maximum(open_, close) * (1.0 + spread)
        low = np.minimum(open_, close) * (1.0 - spread)
        vol = rng.integers(50_000, 400_000, n).astype(float)
        out[sym] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol, "session": "RTH"},
            index=idx,
        )
    return out
