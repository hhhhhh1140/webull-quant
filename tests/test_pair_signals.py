import numpy as np
import pandas as pd
import pytest

from wquant.pair_signals import RTH_RULES, signals


def frame(close, volume=None):
    close = np.asarray(close, dtype=float)
    index = pd.date_range("2026-11-04 09:30", tz="America/New_York", periods=len(close), freq="5min")
    return pd.DataFrame({
        "open": close, "high": close + 0.1, "low": close - 0.1, "close": close,
        "volume": np.ones(len(close)) if volume is None else volume,
    }, index=index)


def test_range_uses_opening_wicks_then_one_confirmed_close_per_side():
    long = frame([100] * 6 + [100.1, 100.5, 101, 99.5, 98, 101])
    sides, targets = signals("opening_range", long, long)
    assert sides.tolist() == [0] * 7 + [1, 0, -1, 0, 0]
    assert np.isnan(targets).all()


@pytest.mark.parametrize("direction", [1, -1])
def test_pullback_requires_ema_warmup_and_confirmed_recovery(direction):
    moves = np.r_[np.arange(24), 16, 18, 25]
    long = frame(100 + direction * moves)
    sides, _ = signals("trend_pullback", long, long)
    assert sides.tolist() == [0] * (len(long) - 1) + [direction]


@pytest.mark.parametrize("long_side", [False, True])
def test_vwap_reclaim_uses_own_etf_volume_and_sets_raw_target(long_side):
    candidate = frame([100] * 6 + [97, 99.2], [2] * 6 + [1, 1])
    other = frame([31] * len(candidate))
    legs = (candidate, other) if long_side else (other, candidate)
    sides, targets = signals("vwap_reclaim", *legs)
    assert sides.tolist() == [0] * 7 + [1 if long_side else -1]
    assert targets[-1] == pytest.approx((1200 + 97 + 99.2) / 14)
    assert np.isnan(targets[:-1]).all()
    tied, targets = signals("vwap_reclaim", candidate, candidate)
    assert not tied.any() and np.isnan(targets).all()


@pytest.mark.parametrize("rule", RTH_RULES)
def test_later_bars_do_not_rewrite_signals_or_targets(rule):
    rng = np.random.default_rng(22)
    long = frame(100 * np.exp(rng.normal(0, 0.012, 48).cumsum()))
    short = frame(60 * 100 / long.close)
    complete, goals = signals(rule, long, short)
    for n in range(1, len(long)):
        prefix, prefix_goals = signals(rule, long.iloc[:n], short.iloc[:n])
        np.testing.assert_array_equal(prefix, complete[:n])
        np.testing.assert_array_equal(prefix_goals, goals[:n])
