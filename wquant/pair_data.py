"""按交易所日历校验 SOXL/SOXS M5；特征只使用信号时刻已经收盘的数据。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from . import data, intraday_data
from .scalp_data import ET, M5, OHLCV, _valid, session_bounds, sessions

SEOUL = "Asia/Seoul"


@dataclass
class Day:
    date: date
    L: pd.DataFrame
    S: pd.DataFrame
    has_night: bool
    has_pre: bool
    prev_close: float | None
    bear: bool | None
    korea_move: pd.Series
    has_korea_history: bool
    pre_signal_close: float | None
    night_signal_close: float | None
    korea_at_four: float
    sector_return_3d: float = np.nan


def _read(path: Path, *, daily_zone: str | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, index_col=0)
    stamps = frame.index.astype(str)
    if daily_zone and stamps.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        frame.index = pd.to_datetime(stamps)
    else:
        if not stamps.str.contains(r"(?:Z|[+-]\d\d:?\d\d)$").all():
            raise ValueError(f"{path.name} 的分钟线或日线时间戳需要时区")
        index = pd.to_datetime(stamps, utc=True, format="mixed").tz_convert(daily_zone or ET)
        frame.index = index.tz_localize(None).normalize() if daily_zone else index
    return frame.sort_index()


def _close(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    closes = pd.to_numeric(frame.close, errors="coerce")
    return closes.where(np.isfinite(closes) & (closes > 0))


def _merge(old: pd.Series, new: pd.Series) -> pd.Series:
    combined = pd.concat([old, new])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def _sector_features(closes: pd.Series):
    if len(closes) < 4:
        return {}, {}
    calendar = xcals.get_calendar("XNYS", start=closes.index.min(), end=closes.index.max())
    prices = closes.reindex(calendar.sessions)
    ma = prices.rolling(20).mean().dropna()
    moves = prices.pct_change(3, fill_method=None).where(prices.rolling(4).count() == 4)
    return (prices.loc[ma.index] < ma).to_dict(), moves.dropna().to_dict()


def _korea_inputs(directory: Path, external: bool):
    daily = _close(_read(directory / "KOSPI_1d.csv", daily_zone=SEOUL))
    observations, spans = [], []
    for interval in ([5] if external else [60, 5]):
        name = "1h" if interval == 60 else "5m"
        closes = _close(_read(directory / f"KOSPI_{name}.csv")).dropna()
        if closes.empty:
            continue
        if external:  # Yahoo 导入使用开盘标签，本地原有 KOSPI 缓存使用收盘标签。
            closes.index = closes.index + M5
        closes = closes.loc[closes.index == closes.index.floor("min")]
        if closes.empty:
            continue
        seoul = closes.index.tz_convert(SEOUL)
        spans.append((seoul[0].date(), seoul[-1].date()))
        observations.append(pd.DataFrame({"price": closes, "minutes": interval}))
    return daily, observations, spans


def _korea_series(daily, observations):
    if daily.empty or not observations:
        return pd.DataFrame()
    quotes = pd.concat(observations)
    quotes = quotes.loc[~quotes.index.duplicated(keep="last")].sort_index()
    dates = quotes.index.tz_convert(SEOUL).tz_localize(None).normalize()
    cal = xcals.get_calendar("XKRX", start=dates.min() - pd.Timedelta(days=21), end=dates.max())
    prev = daily.reindex(cal.sessions).shift()
    quotes["move"] = quotes.price / prev.reindex(dates).to_numpy() - 1
    return quotes


def _moves(quotes: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    if quotes.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    signals = (index + M5).tz_convert(SEOUL)
    loc = quotes.index.get_indexer(signals, method="pad")
    recent = quotes.iloc[loc.clip(min=0)]
    stamped = recent.index.tz_convert(SEOUL)
    # 收盘后沿用本韩国交易日最后的分时价；交易中不沿用过期的 K 线。
    after_close = (signals.hour * 60 + signals.minute) >= 15 * 60 + 30
    near_close = (stamped.hour * 60 + stamped.minute) >= 15 * 60
    fresh = signals - stamped <= pd.to_timedelta(recent.minutes.to_numpy(), unit="min")
    valid = (loc >= 0) & (signals.date == stamped.date) & (fresh | (after_close & near_close))
    return pd.Series(np.where(valid, recent.move.to_numpy(), np.nan), index=index)


def _segment(L, S, start, end, label):
    expected = pd.date_range(start, end, freq=M5, inclusive="left")
    pair = []
    for frame in (L, S):
        part = frame.loc[start:end - M5]
        part = part.loc[part.session == label, OHLCV + ["session"]]
        if not _valid(part[OHLCV], expected):
            return None
        pair.append(part)
    return pair


def _single_close(frame, stamp, label):
    part = frame.loc[stamp:stamp]
    if len(part) != 1 or part.session.iat[0] != label:
        return None
    o, h, lo, c = part[["open", "high", "low", "close"]].iloc[0].astype(float)
    if np.isfinite([o, h, lo, c]).all() and 0 < lo <= min(o, c) <= max(o, c) <= h:
        return float(c)
    return None


def _days(L, S, anchors, flags, quotes, spans, after=None, sector_moves=None):
    known = quotes.dropna(subset=["move"]) if not quotes.empty else quotes
    if known.empty:
        spans = []
    else:
        first, last = known.index[[0, -1]].tz_convert(SEOUL).date
        spans = [(max(first, lo), min(last, hi)) for lo, hi in spans]
    result = []
    for day, left, right, _ in sessions(L, S):
        if after and day <= after:
            continue
        opening = left.index[0]
        pre = opening.replace(hour=4, minute=0)
        night = pd.Timestamp(day - timedelta(days=1), tz=ET).replace(hour=20)
        ovn = _segment(L, S, night, pre, "OVN")
        pm = _segment(L, S, pre, opening, "PRE")
        segments = [p for p in (ovn, pm) if p is not None]
        segments.append([left.assign(session="RTH"), right.assign(session="RTH")])
        long, short = (pd.concat(parts) for parts in zip(*segments))
        prev = day - timedelta(days=1)
        while (bounds := session_bounds(prev)) is None:
            prev -= timedelta(days=1)
        prior = _single_close(anchors, bounds[1] - M5, "RTH")
        signal_bar = pre - M5
        moves = _moves(quotes, long.index.union(pd.DatetimeIndex([signal_bar])))
        result.append(Day(day, long, short, ovn is not None, pm is not None, prior,
                          flags.get(pd.Timestamp(prev)), moves.reindex(long.index),
                          any(start <= day <= end for start, end in spans),
                          _single_close(L, opening - 4 * M5, "PRE"),
                          _single_close(L, signal_bar, "OVN"), float(moves.at[signal_bar]),
                          (sector_moves or {}).get(pd.Timestamp(prev), np.nan)))
    return result


def study_days(forward_dir=None) -> list[Day]:
    """历史采用原始 Webull M5，外部样本从其最后一个完整 RTH 日之后起算。

    外部 SOXL_M5、SOXS_M5、KOSPI_5m CSV 的索引均为带时区的开盘时刻；
    SOXX_1d 和 KOSPI_1d 的索引使用交易所本地日期或带时区的日线时间戳。
    """
    L, S = (intraday_data.load(symbol, adjust=False) for symbol in ("SOXL", "SOXS"))
    trend = _close(_read(data.CACHE / "SOXX.csv", daily_zone=ET))
    daily, obs, spans = _korea_inputs(intraday_data.DIR, False)
    flags, sector_moves = _sector_features(trend)
    days = _days(L, S, L, flags, _korea_series(daily, obs), spans, sector_moves=sector_moves)
    if forward_dir is None:
        return days
    folder = Path(forward_dir)
    paths = [folder / f"{sym}_M5.csv" for sym in ("SOXL", "SOXS")]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    fresh_L, fresh_S = map(_read, paths)
    last = days[-1].date if days else None
    # 共享前一日的收盘锚点。外部行情不改变历史样本的数据。
    new = fresh_L.loc[fresh_L.index.date > last] if last else fresh_L
    anchors = pd.concat([L.loc[~L.index.isin(new.index)], new]).sort_index()
    trend_path = folder / "SOXX_1d.csv"
    if not trend_path.exists():
        trend_path = folder / "SOXX_daily.csv"
    trend = _merge(trend, _close(_read(trend_path, daily_zone=ET)))
    new_daily, new_obs, new_spans = _korea_inputs(folder, True)
    flags, sector_moves = _sector_features(trend)
    return days + _days(fresh_L, fresh_S, anchors, flags,
                        _korea_series(_merge(daily, new_daily), obs + new_obs),
                        spans + new_spans, after=last, sector_moves=sector_moves)
