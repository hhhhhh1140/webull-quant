import numpy as np
import pandas as pd

from wquant import pair_data as pair


def candles(index, label, price=50.0):
    return pd.DataFrame({"open": price, "high": price + 1, "low": price - 1,
                         "close": price, "volume": 100.0, "session": label}, index=index)


def eastern(t):
    return pd.Timestamp(t, tz=pair.ET)


def test_segments_prior_half_day_close_and_fixed_premarket_quote():
    # Friday after Thanksgiving was an early close; the Monday night is Sunday.
    rth = pd.date_range(eastern("2025-12-01 09:30"), periods=78, freq="5min")
    pre = pd.date_range(eastern("2025-12-01 04:00"), periods=66, freq="5min")
    night = pd.date_range(eastern("2025-11-30 20:00"), periods=96, freq="5min")
    last = candles([eastern("2025-11-28 12:55")], "RTH", price=45.)
    left = pd.concat([last, candles(night, "OVN"), candles(pre, "PRE"), candles(rth, "RTH")])
    right = left.drop(night[5])
    day = pair._days(left, right, left, {}, pd.DataFrame(), [])[0]
    assert not day.has_night and day.has_pre and day.prev_close == 45
    assert day.night_signal_close == 50
    assert day.pre_signal_close == 50 and day.L.index.equals(day.S.index)
    assert set(day.L.session) == {"PRE", "RTH"} and day.bear is None
    korean = pd.DataFrame({"minutes": [5, 5], "move": [.02, -.04]}, index=pre[:2])
    incomplete_night = pair._days(left, right, left, {}, korean, [])[0]
    assert incomplete_night.korea_at_four == .02
    assert incomplete_night.korea_move.index.equals(incomplete_night.L.index)

    # Public PRE has zero volume, but its exact 09:10 OHLC can be an input to an RTH fill.
    left.loc[pre, "volume"] = 0
    day = pair._days(left, right, left.drop(last.index), {}, pd.DataFrame(), [])[0]
    assert not day.has_pre and day.prev_close is None and day.pre_signal_close == 50
    assert day.korea_move.isna().all() and day.L.index.equals(rth)
    day = pair._days(left.drop(eastern("2025-12-01 09:10")), right, left, {}, pd.DataFrame(), [])[0]
    assert day.pre_signal_close is None


def test_korean_observation_available_at_bar_close_and_same_day_only(tmp_path):
    # First five-minute bar opens 09:00 Seoul and becomes knowable at 09:05.
    stocks = pd.DataFrame({"close": [100., 200.]}, index=["2025-12-01", "2025-12-02"])
    bars = pd.DataFrame({"close": [103., 105.]},
                        index=pd.to_datetime(["2025-12-02 09:00", "2025-12-02 09:05"]).tz_localize(pair.SEOUL))
    stocks.to_csv(tmp_path / "KOSPI_1d.csv")
    bars.to_csv(tmp_path / "KOSPI_5m.csv")
    daily, obs, spans = pair._korea_inputs(tmp_path, external=True)
    assert spans == [(bars.index[0].date(), bars.index[0].date())]
    signals = pd.DatetimeIndex([bars.index[0] - pair.M5, *bars.index, bars.index[-1] + 5 * pair.M5,
                                bars.index[0] + pd.Timedelta(days=1)]).tz_convert(pair.ET)
    moves = pair._moves(pair._korea_series(daily, obs), signals)
    np.testing.assert_allclose(moves, [np.nan, .03, .05, np.nan, np.nan], equal_nan=True)
