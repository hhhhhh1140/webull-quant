from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wquant.pair_study import Entry, eligible, ideas, simulate


def day_fixture(bars, sessions=None):
    frame = pd.DataFrame(bars, columns=["open", "high", "low"],
                         index=pd.date_range("2026-10-07 09:30", periods=len(bars),
                                             freq="5min", tz="America/New_York"))
    frame["close"] = frame.open
    frame["session"] = sessions or "RTH"
    return SimpleNamespace(L=frame, S=frame.copy(), date=date(2026, 10, 7),
                           prev_close=100, pre_signal_close=96, has_pre=False, has_night=False)


def test_pre_gap_only_uses_0910_and_prior_close():
    d = day_fixture([[100, 101, 99]] * 3)
    assert eligible(d, "pre_gap") and not eligible(d, "at_four")
    entry, = ideas(d, "pre_gap")
    assert (entry.bar, entry.side, entry.last) == (0, -1, 2)
    d.pre_signal_close = np.nan
    assert not eligible(d, "pre_gap")


@pytest.mark.parametrize("night_close,expected_side", [(103.1, 1), (96.9, -1), (101, 0)])
def test_at_four_uses_last_night_bar_even_if_an_earlier_bar_is_missing(night_close, expected_side):
    d = day_fixture([[100, 101, 99]] * 4, ["PRE", "PRE", "RTH", "RTH"])
    d.has_pre = True
    d.night_signal_close = night_close
    assert eligible(d, "at_four")
    candidates = ideas(d, "at_four")
    assert [e.side for e in candidates] == ([expected_side] if expected_side else [])
    assert all(e.bar == 0 and e.last == 3 for e in candidates)
    d.night_signal_close = None
    assert not eligible(d, "at_four")


def test_stop_first_and_roundtrip_costs_on_actual_leg():
    d = day_fixture([[30, 31, 29]] * 4)
    d.S.loc[:, ["open","high","low"]] = [[100,110,94], [100,101,99], [102,103,101], [99,99,98]]
    t, = simulate(d, [Entry(0,-1,3)], 3000, 1)
    assert (t.symbol, t.qty, t.exit_time) == ("SOXS",9,d.S.index[0])
    assert t.pnl == pytest.approx(9 * (97 * .9995 - 100 * 1.0005) - .2)
    assert not simulate(d, [Entry(0,-1,3, target=100.08)], 3000, 1)


def test_trailing_high_only_affects_next_bar_and_gap_exits_at_open():
    d = day_fixture([[99,110,98], [97,101,96], [90,92,89], [94,95,93]],
                    ["OVN","RTH","RTH","RTH"])
    t, = simulate(d, [Entry(0, 1, 3, stop=.06, trail=.07, take_pct=np.inf)], 3000, 1)
    assert t.entry_price == pytest.approx(99 * 1.0015)
    assert t.exit_time == d.L.index[1]
    assert t.exit_price == pytest.approx(97 * .9995)
