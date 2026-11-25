from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from wquant.scalp import ScalpParams, run_session


DAY = "2026-11-04"
TZ = "America/New_York"
CLOSE = pd.Timestamp(f"{DAY} 16:00", tz=TZ)
BASE = ScalpParams(slippage_bps=0, spread_bps=0, fee_per_order=0)


def candles(prices):
    close = np.array(prices, dtype=float)
    opening = np.r_[close[0], close[:-1]]
    idx = pd.date_range(f"{DAY} 09:30", periods=len(close), freq="5min", tz=TZ)
    return pd.DataFrame({"open": opening, "high": np.maximum(opening, close),
                         "low": np.minimum(opening, close), "close": close}, index=idx)


def pair(prices):
    return candles(prices), candles(10000 / np.array(prices, dtype=float))


def test_confirm_next_open_roundtrips_both_etfs_and_partial_replay():
    long, short = pair([100] * 6 + [98, 98.5, 100] + [100] * 8 + [102, 101.5, 100])
    early_trades, early = run_session(long.iloc[:8], short.iloc[:8], BASE, CLOSE)
    assert early_trades == [] and early["qty"] == 0 and early["pending"] == "BUY SOXL"
    trades, state = run_session(long, short, BASE, CLOSE)
    assert [t["symbol"] for t in trades] == ["SOXL", "SOXS"]
    assert [t["entry_t"] for t in trades] == [long.index[8].isoformat(), long.index[-1].isoformat()]
    assert [t["reason"] for t in trades] == ["止盈", "止盈"]
    assert trades[1]["entry"] == pytest.approx(short.open.iat[-1])
    assert state["equity"] == pytest.approx(BASE.capital + sum(t["pnl"] for t in trades))
    late_trades, _ = run_session(long, short, BASE, CLOSE, trade_from=long.index[8])
    assert late_trades == trades[1:]
    future_trades, future = run_session(long, short, BASE, CLOSE, trade_from=CLOSE)
    assert future_trades == [] and future["entries"] == 0


def test_stop_precedes_target_and_gap_gets_adverse_open():
    long, short = pair([100] * 6 + [98, 98.5, 98.5, 98.5])
    long.iloc[-1] = [98.5, 105, 90, 98.5]
    trades, _ = run_session(long, short, BASE, CLOSE)
    assert trades[-1]["reason"] == "止损" and trades[-1]["exit"] == pytest.approx(98.5 * 0.992)
    long.iloc[-1] = [89.0, 105, 88, 98.5]
    trades, _ = run_session(long, short, BASE, CLOSE)
    assert trades[-1]["exit"] == 89


def test_cash_costs_day_loss_and_trade_count_cap():
    long, short = pair([100] * 6 + [98, 98.5, 98.5, 98.5] + [100] * 10 + [102, 101.5, 100])
    long.iloc[9] = [96, 98.6, 95, 98.5]
    p = ScalpParams(capital=300, notional=1000, daily_loss_pct=0.01)
    trades, state = run_session(long, short, p, CLOSE)
    first = trades[0]
    assert first["qty"] == 3
    assert first["exit"] == pytest.approx(96 * (1 - p.fill_drag))
    assert first["costs"] == pytest.approx(2 * p.fee_per_order + 3 * (98.5 + 96) * p.fill_drag)
    assert state["pnl"] == pytest.approx(3 * (96 - 98.5) - first["costs"])
    assert state["cash"] == pytest.approx(300 + state["pnl"])
    assert state["halted"] and len(trades) == 1
    trades, state = run_session(long, short, replace(p, daily_loss_pct=0.9, max_trades=1), CLOSE)
    assert len(trades) == 1 and not state["halted"]


def test_partial_day_does_not_force_exit_and_final_bar_sells_at_open():
    long, short = pair([100] * 6 + [98, 98.5, 98.5, 98.5, 98.5])
    early_close = long.index[-1] + pd.Timedelta(minutes=5)
    long.iloc[-1] = [98.4, 102, 90, 100.1]
    trades, state = run_session(long.iloc[:-1], short.iloc[:-1], BASE, early_close)
    assert trades == [] and state["symbol"] == "SOXL" and state["qty"] == 10
    trades, state = run_session(long, short, BASE, early_close)
    assert trades[-1]["exit"] == 98.4 and trades[-1]["reason"] == "收盘平仓"
    assert state["qty"] == 0 and state["pending"] is None


def test_hold_limit_and_no_entries_on_final_bar():
    long, short = pair([100] * 6 + [98, 98.5, 98.5, 98.55, 98.6, 98.6])
    p = replace(BASE, max_hold_bars=2)
    trades, state = run_session(long.iloc[:10], short.iloc[:10], p, CLOSE)
    assert state["pending"] == "SELL" and trades == []
    trades, _ = run_session(long, short, p, CLOSE)
    assert trades[0]["exit_t"] == long.index[10].isoformat() and trades[0]["reason"] == "持仓超时"
    trades, state = run_session(long.iloc[:9], short.iloc[:9], p, long.index[9])
    assert not trades and state["entries"] == 0


def test_mismatched_times_and_missing_bar_rejected():
    long, short = pair([100] * 12)
    with pytest.raises(ValueError, match="时间不一致"):
        run_session(long, short.iloc[:-1], BASE, CLOSE)
    with pytest.raises(ValueError, match="连续"):
        run_session(long.drop(long.index[7]), short.drop(short.index[7]), BASE, CLOSE)


def test_mark_to_market_loss_stops_entries_and_liquidates_next_open():
    long, short = pair([100] * 6 + [98, 98.5, 98.5, 95, 90, 98] + [100] * 8 + [102, 101.5, 100])
    p = replace(BASE, stop_pct=0.1, daily_loss_pct=0.01)
    prefix_trades, prefix = run_session(long.iloc[:10], short.iloc[:10], p, CLOSE)
    assert not prefix_trades and prefix["halted"] and prefix["pending"] == "SELL"
    trades, state = run_session(long, short, p, CLOSE)
    assert len(trades) == 1 and trades[0]["reason"] == "日亏损"
    assert trades[0]["exit"] == 95 and state["qty"] == 0 and state["halted"]


def test_reversion_sells_next_open_on_traded_leg():
    long, short = pair([100] * 6 + [102, 101.5, 100.8, 99.9, 99.5])
    trades, _ = run_session(long, short, replace(BASE, target_pct=0.10), CLOSE)
    assert trades[0]["reason"] == "回归均值" and trades[0]["symbol"] == "SOXS"
    assert trades[0]["exit"] == pytest.approx(short.open.iat[10])
