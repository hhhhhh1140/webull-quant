from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wquant.night_pullback import entries, replay_pullback
from wquant.pair_data import _sector_features
from wquant.scalp_data import ET


def night():
    index = pd.date_range("2026-07-06 20:00", periods=96, freq="5min", tz=ET)
    bars = pd.DataFrame({"open": 100., "high": 100., "low": 100., "close": 100.,
                         "session": "OVN", "volume": 300.}, index=index)
    return SimpleNamespace(L=bars, S=bars.copy(), date=index[-1].date(), has_night=True,
                           sector_return_3d=.02, prev_close=100.)


@pytest.mark.parametrize("side", [1, -1])
def test_direction_uses_pullback_of_selected_etf_at_last_close(side):
    day = night()
    day.sector_return_3d *= side
    chosen = day.L if side == 1 else day.S
    chosen.loc[chosen.index[11:14], ["open", "high", "low", "close"]] = 99.
    order = entries(day, "deep")[0]
    assert order.side == side and order.bar == 12 and order.last == 24
    assert order.max_buy_price == pytest.approx((1100 + 99) / 12 * .997)
    assert entries(day, "vwap_exit")[0].target == pytest.approx((1100 + 99) / 12)
    assert entries(day, "deep", delay=1)[0].bar == 13
    assert entries(day, "confirmed") == []
    day.sector_return_3d = .004
    assert entries(day, "both") == []


def test_frozen_limit_skips_gap_and_delay_stays_in_night():
    day = night()
    day.L.loc[day.L.index[11], ["open", "high", "low", "close"]] = 99.
    day.L.loc[day.L.index[13], ["open", "high", "low", "close"]] = 98.
    _, trades = replay_pullback([day], "deep")
    assert trades == []
    _, later = replay_pullback([day], "deep", delay=1)
    assert later[0].entry_time == day.L.index[13] and later[0].entry_price < 99
    day.L.loc[day.L.index[-2], ["open", "high", "low", "close"]] = 99.
    assert all(e.bar < e.last <= 95 for e in entries(day, "both", delay=1))


def test_sector_return_uses_three_exchange_sessions_and_requires_all_four_closes():
    prices = pd.Series([100., 103., 101., 104., 99.], index=pd.to_datetime(
        ["2026-07-01", "2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08"]))
    _, moves = _sector_features(prices)
    assert moves[prices.index[3]] == pytest.approx(.04)
    prices.iloc[2] = np.nan
    assert _sector_features(prices)[1] == {}
