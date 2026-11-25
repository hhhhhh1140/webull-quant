"""固定美东时点的方向交易。输入为 pair_data.study_days 已校验的 M5 数据。"""
from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from numbers import Integral

import numpy as np
import pandas as pd

from .pair_data import Day
from .pair_study import Entry
from .scalp_data import ET, M5, session_bounds


def _window(day: Day, clock: str, delay: int):
    if not isinstance(clock, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5][05]", clock):
        raise ValueError("clock 须为美东 HH:MM，分钟为 5 的倍数")
    if isinstance(delay, bool) or not isinstance(delay, Integral) or delay < 0:
        raise ValueError("delay 须为非负整数，单位为 5 分钟线")
    when = time.fromisoformat(clock)
    local_day = day.date - timedelta(days=1) if when.hour >= 20 else day.date
    planned = pd.Timestamp(datetime.combine(local_day, when), tz=ET)
    bounds = session_bounds(day.date)
    if bounds is None or pd.isna(day.prev_close) or not np.isfinite(day.prev_close) or day.prev_close <= 0:
        return None
    final = bounds[1] - M5
    fill = planned + int(delay) * M5
    L, S = day.L, day.S
    idx = L.index
    if (fill >= final or not idx.equals(S.index) or not isinstance(idx, pd.DatetimeIndex)
            or idx.tz is None or not idx.is_unique or not idx.is_monotonic_increasing):
        return None
    # simulate 以相同整数位置读取两只 ETF；持仓段的每个时点均须存在。
    first, last = idx.get_indexer([planned, final])
    if first < 0 or last < 0:
        return None
    path = idx[first:last + 1]
    if not path.equals(pd.date_range(planned, final, freq=M5).tz_convert(idx.tz)):
        return None
    previous = planned - M5
    if first and idx[first - 1] == previous:
        signal = L.close.iat[first - 1]
    elif clock == "04:00":
        # 当天有完整盘前线，但整段夜盘可能因更早的数据缺口而被丢弃。
        signal = day.night_signal_close
    else:
        return None
    if pd.isna(signal) or not np.isfinite(signal) or signal <= 0:
        return None
    return int(first + delay), int(last), signal / day.prev_close - 1


def eligible_clock(day: Day, clock: str, *, delay: int = 0) -> bool:
    """时点有可校验的信号与成对持仓行情；无论是否达到开仓阈值均纳入样本。"""
    return _window(day, clock, delay) is not None


def clock_entry(day: Day, clock: str, *, delay: int = 0) -> Entry | None:
    """用计划入场时点的上一根 SOXL 收盘与上一常规盘收盘比较；延迟仅移动成交时刻。"""
    window = _window(day, clock, delay)
    if window is None:
        return None
    bar, last, move = window
    if abs(move) < .03:
        return None
    return Entry(bar, int(np.sign(move)), last, stop=.06, take_pct=np.inf, trail=.07)
