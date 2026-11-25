import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wquant.clock_study import replay_clock
from wquant.pair_sizing import load_signals
from wquant.sizing_news import NewsBook
from wquant.entry_clock import clock_entry, eligible_clock
from wquant.scalp_data import ET, M5, session_bounds


def day_data(day=date(2026, 11, 9)):
    opening, end = session_bounds(day)
    idx = pd.date_range(opening.normalize() - pd.Timedelta(hours=4), end, freq=M5, inclusive="left")
    minute = idx.hour * 60 + idx.minute
    labels = np.select([minute < 240, minute < 570, minute < 960], ["OVN", "PRE", "RTH"], default="OVN")
    bars = pd.DataFrame({"session": labels, "open": 100., "high": 102., "low": 95., "close": 100.,
                         "volume": 1000.}, index=idx)
    return SimpleNamespace(date=day, L=bars, S=bars.copy(), prev_close=100., night_signal_close=102.)


def stamp(day, clock):
    return pd.Timestamp(f"{day.date} {clock}", tz=ET)


@pytest.mark.parametrize(("clock", "signal_clock", "prev_day"), [
    ("21:00", "20:55", True), ("00:00", "23:55", True), ("04:00", "03:55", False),
    ("09:30", "09:25", False), ("09:35", "09:30", False),
])
def test_exact_clock_and_preceding_closed_bar(clock, signal_clock, prev_day):
    d = day_data()
    prior = stamp(d, signal_clock) - pd.Timedelta(days=int(prev_day))
    d.L.loc[prior, "close"] = 96.
    d.L.loc[prior + M5, "close"] = 101.
    assert eligible_clock(d, clock)
    e = clock_entry(d, clock)
    assert e.side == -1 and d.L.index[e.bar] == prior + M5
    assert d.L.index[e.last] == stamp(d, "15:55")
    assert e.stop == .06 and e.trail == .07 and np.isinf(e.take_pct)
    delayed = clock_entry(d, clock, delay=1)
    assert delayed.side == e.side and d.L.index[delayed.bar] == prior + 2 * M5


def test_four_can_use_single_confirmed_night_bar_if_full_overnight_omitted():
    d = day_data()
    d.L = d.L.loc[stamp(d, "04:00"):]
    d.S = d.S.loc[d.L.index]
    d.night_signal_close = 104.
    assert clock_entry(d, "04:00").side == 1
    d.night_signal_close = None
    assert not eligible_clock(d, "04:00")


def test_neutral_signal_and_true_close_of_short_session():
    d = day_data(date(2026, 11, 27))
    assert eligible_clock(d, "09:30") and clock_entry(d, "09:30") is None
    d.L.loc[stamp(d, "09:25"), "close"] = 96.
    e = clock_entry(d, "09:30")
    assert d.L.index[e.last] == stamp(d, "12:55")
    assert not eligible_clock(d, "12:50", delay=1)
    assert not eligible_clock(d, "13:00")


def test_missing_signal_or_path_prevents_entry():
    d = day_data()
    d.L = d.L.drop(index=stamp(d, "09:25"))
    d.S = d.S.loc[d.L.index]
    assert not eligible_clock(d, "09:30")
    assert eligible_clock(d, "09:35")
    d.S = d.S.drop(index=stamp(d, "12:25"))
    assert not eligible_clock(d, "09:35")
    d.L = d.L.loc[d.S.index]
    assert not eligible_clock(d, "09:35")


@pytest.mark.parametrize(("clock", "delay"), [("9:30", 0), ("09:03", 0), ("25:00", 0), ("09:30", -1)])
def test_invalid_clock_arguments(clock, delay):
    with pytest.raises(ValueError):
        clock_entry(day_data(), clock, delay=delay)


def test_deferred_fill_keeps_earlier_direction_and_cannot_precede_signal():
    day = day_data()
    day.L.loc[stamp(day, "03:55"), ["high", "close"]] = [105., 104.]
    day.L.loc[stamp(day, "09:25"), "close"] = 96.

    _, deferred = replay_clock([day], "04:00", defer_to="09:30")
    _, fresh = replay_clock([day], "09:30")
    assert len(deferred) == len(fresh) == 1
    assert deferred[0].entry_time == fresh[0].entry_time == stamp(day, "09:30")
    assert deferred[0].symbol == "SOXL" and fresh[0].symbol == "SOXS"
    with pytest.raises(ValueError, match="信号确认后"):
        replay_clock([day], "09:30", defer_to="04:00")

    day.L.loc[stamp(day, "03:55"), "close"] = 100.
    assert replay_clock([day], "04:00", defer_to="09:30") == ([0.0], [])


def test_sizing_later_clock_uses_korea_and_news_as_known_at_that_clock(tmp_path):
    day = day_data()
    cutoff = stamp(day, "09:30")
    day.L.loc[cutoff - M5, ["low", "close"]] = [92., 93.]
    day.korea_at_four = .05
    day.korea_move = pd.Series(np.nan, index=day.L.index)
    day.korea_move.at[cutoff - M5] = -.02
    day.korea_move.at[cutoff] = .04  # 09:30 这根的值到 09:35 才可见。

    archive = tmp_path / "news.jsonl"
    known = {"headline": "测试消息 A", "source": "fixture", "scope": "semis", "direction": -1,
             "published_at": (cutoff - M5).isoformat(), "available_at": cutoff.isoformat()}
    late = {**known, "headline": "测试消息 B", "direction": 1,
            "available_at": (cutoff + pd.Timedelta(seconds=1)).isoformat()}
    archive.write_text("\n".join(map(json.dumps, (known, late))), encoding="utf-8")
    earnings = tmp_path / "earnings.csv"
    earnings.write_text("issuer,published_at,source\n", encoding="utf-8")
    signal, = load_signals([day], NewsBook.load(archive), earnings, clock="09:30")

    assert signal.order.side == -1 and signal.move == pytest.approx(-.07)
    assert signal.korea == -.02 and signal.strong
    assert signal.news.headlines == 1 and signal.news.confirms(-1)
