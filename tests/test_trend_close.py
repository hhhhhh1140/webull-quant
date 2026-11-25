import numpy as np
import pandas as pd
import pytest

from wquant import trend_close
from wquant.scalp_data import ET, M5


def session():
    idx = pd.date_range("2026-11-26 20:00", "2026-11-27 12:55", freq=M5, tz=ET)
    labels = np.select([idx < pd.Timestamp("2026-11-27 04:00", tz=ET),
                        idx < pd.Timestamp("2026-11-27 09:30", tz=ET)], ["OVN", "PRE"], default="RTH")
    bars = pd.DataFrame({"open": 100., "high": 100., "low": 100., "close": 100.,
                         "volume": 1000., "session": labels}, index=idx)
    return trend_close.CloseDay(idx[-1].date(), .02, bars, bars.copy(), 0)


@pytest.mark.parametrize(("trend", "sym"), [(.05, "SOXL"), (-.05, "SOXS"), (.049, None), (-.049, None)])
def test_direction_and_first_fill_held_through_drawdown_to_early_close(trend, sym):
    day = session()
    day.trend = trend
    cols = ["open", "high", "low", "close"]
    day.L.loc[day.L.index[11:13], cols] = 99 if trend > 0 else 101
    day.S.loc[day.S.index[11:13], cols] = 99
    chosen = day.L if trend > 0 else day.S
    chosen.loc[chosen.index[18], cols] = 75
    chosen.loc[chosen.index[-1], cols] = 110
    _, trades, dd = trend_close.replay_close([day])
    if sym is None:
        assert not trades and dd == 0
    else:
        t, = trades
        assert (t.symbol, t.entry_time, t.exit_time) == (sym, chosen.index[12], chosen.index[-1])
        assert dd < -.06 and t.pnl > 0


def test_actual_early_close_print_required_but_missing_intermediate_mark_allowed(monkeypatch):
    day = session()
    sector = pd.Series([98., 99., 100., 104.], index=pd.to_datetime(
        ["2026-11-20", "2026-11-23", "2026-11-24", "2026-11-25"]))
    day.L = day.L.drop(day.L.index[-2])
    monkeypatch.setattr(trend_close, "market_sources", lambda _: [(day.L, day.S, sector, None)])
    d, = trend_close.close_days(None)
    assert d.trend == pytest.approx(104/98 - 1) and d.missing_marks == 1
    assert d.L.index[-1].strftime("%H:%M") == "12:55"
    assert trend_close.close_days(None, exit_bars_before=2) == []


def test_bearish_vwap_entry_does_not_require_bull_etf_to_rally():
    day = session()
    day.trend = -.06
    day.S.loc[day.S.index[11:13], ["open", "high", "low", "close"]] = 99.
    assert not trend_close.replay_close([day])[1]
    t, = trend_close.replay_close([day], require_market_move=False)[1]
    assert t.symbol == "SOXS" and t.exit_time == day.S.index[-1]


@pytest.mark.parametrize("threshold", [.01, .03, .05])
def test_threshold_includes_boundary_on_either_side(threshold):
    for sign in (-1, 1):
        move = (100 + sign * threshold * 100) / 100 - 1
        assert trend_close.trend_side(move, threshold) == sign
        assert trend_close.trend_side(move - sign * .000001, threshold) == 0
