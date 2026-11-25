"""Regular-session Webull M5 bars. Each timestamp is the *start* of the bar."""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Iterator

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from .intraday_data import _to_frame
from .webull_client import _json, _unwrap_list

ET = "America/New_York"
M5 = pd.Timedelta(minutes=5)
OHLCV = ["open", "high", "low", "close", "volume"]


@lru_cache(maxsize=8)
def _calendar(year: int):
    return xcals.get_calendar("XNYS", start=f"{year}-01-01", end=f"{year}-12-31")


def session_bounds(day: date) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """NYSE regular trading hours in New York time, including early closes."""
    calendar = _calendar(day.year)
    key = pd.Timestamp(day)
    if key not in calendar.sessions:
        return None
    return calendar.session_open(key).tz_convert(ET), calendar.session_close(key).tz_convert(ET)


def _regular_days(frame: pd.DataFrame):
    if frame.empty:
        return {}
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("分钟线必须使用带时区的时间索引")
    rth = frame.loc[frame.session == "RTH", OHLCV].tz_convert(ET).sort_index()
    return {day: bars for day, bars in rth.groupby(rth.index.date)}


def _valid(frame: pd.DataFrame, expected: pd.DatetimeIndex) -> bool:
    if not frame.index.equals(expected):
        return False
    values = frame.to_numpy(dtype=float)
    o, h, lo, c, _ = values.T
    return bool(np.isfinite(values).all() and (values > 0).all()
                and (lo <= np.minimum(o, c)).all() and (h >= np.maximum(o, c)).all())


def sessions(L: pd.DataFrame, S: pd.DataFrame, *, as_of=None) -> Iterator[
    tuple[date, pd.DataFrame, pd.DataFrame, pd.Timestamp]
]:
    """Yield aligned RTH days with a complete M5 grid and valid OHLCV in both legs.

    A supplied timezone-aware as_of admits the current session only through
    the latest closed bar. Holidays, truncated and stale days are skipped.
    """
    cutoff = pd.Timestamp.now(tz=ET) if as_of is None else pd.Timestamp(as_of)
    if cutoff.tz is None:
        raise ValueError("as_of 必须带时区")
    cutoff = cutoff.tz_convert(ET)
    longs, shorts = _regular_days(L), _regular_days(S)
    for day in sorted(longs.keys() & shorts.keys()):
        if day > cutoff.date():
            break
        bounds = session_bounds(day)
        if bounds is None:
            continue
        start, close = bounds
        if cutoff < close and as_of is None:
            continue
        end = min(close, cutoff.floor("5min"))
        if end <= start:
            continue
        expected = pd.date_range(start, end, freq=M5, inclusive="left")
        left, right = longs[day], shorts[day]
        left = left.loc[(left.index >= start) & (left.index < end)]
        right = right.loc[(right.index >= start) & (right.index < end)]
        if _valid(left, expected) and _valid(right, expected):
            yield day, left, right, close


def latest_pair(client) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch enough RTH bars for today's pair in one Webull market-data request."""
    response = client.data.market_data.get_batch_history_bar(
        ["SOXL", "SOXS"], "US_STOCK", "M5", count="100",
        trading_sessions="RTH", real_time_required=True,
    )
    rows = {item["symbol"]: item["result"] for item in _unwrap_list(_json(response))
            if item.get("symbol") and item.get("result")}
    if not rows.get("SOXL") or not rows.get("SOXS"):
        raise ValueError("Webull 未返回完整的 SOXL / SOXS 行情")
    return _to_frame(rows["SOXL"]), _to_frame(rows["SOXS"])
