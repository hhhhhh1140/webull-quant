import json

import numpy as np
import pandas as pd
import pytest

from wquant import scalp_paper as paper
from wquant.scalp import ScalpParams

DAY = "2026-11-04"
TOMORROW = "2026-11-05"
PARAMS = ScalpParams()


def at(clock, day=DAY):
    return pd.Timestamp(f"{day} {clock}", tz="America/New_York")


def pair(day=DAY, dip=6, rebound=100):
    prices = np.full(78, 100.)
    prices[dip:dip + 3] = [98, 98.5, rebound]
    frames = []
    for close in (prices, 10_000 / prices):
        opening = np.r_[close[0], close[:-1]]
        frames.append(pd.DataFrame({
            "open": opening, "high": np.maximum(close, opening),
            "low": np.minimum(close, opening), "close": close,
            "volume": 1000., "session": "RTH",
        }, index=pd.date_range(at("09:30", day), periods=78, freq="5min")))
    return frames


@pytest.fixture
def monitor(tmp_path):
    path = tmp_path / "scalp.json"

    def poll(when, bars):
        return paper._poll_once(lambda: bars, PARAMS, path, lambda: when)

    def record(day=DAY):
        return json.loads(path.read_text())["days"][day]

    return poll, record


def test_late_start_duplicate_poll_and_price_revision(monitor):
    poll, record = monitor
    bars = pair()
    poll(at("10:15:02"), bars)
    warm = record()
    assert warm["state"]["entries"] == 0 and warm["trades"] == []
    assert warm["trade_from"] == at("10:15:02").isoformat()
    poll(at("10:16"), bars)
    assert record() == warm
    bars[1].iloc[2, bars[1].columns.get_loc("volume")] += 1
    poll(at("10:20"), bars)
    assert record()["status"] == "excluded"
    assert record()["last_bar"] == warm["last_bar"]


def test_outage_keeps_unresolved_position_and_carries_only_cash(monitor):
    poll, record = monitor
    bars = pair(rebound=98.5)
    for t in ["10:00", "10:05", "10:10", "10:15"]:
        poll(at(t), bars)
    held = record()
    assert held["state"]["symbol"] == "SOXL" and held["state"]["qty"] > 0
    # 缺失的 ETF bar 不能触发成交，长时间断线也不能回放出后来的止盈。
    poll(at("10:20"), [bars[0], bars[1].drop(at("10:15"))])
    assert record() == held
    message = poll(at("10:26"), bars)
    assert "SOXL" in message
    assert record()["status"] == "excluded"
    assert record()["state"] == held["state"] and record()["trades"] == []
    poll(at("10:00", TOMORROW), pair(TOMORROW))
    assert record(TOMORROW)["opening_capital"] == held["state"]["cash"] < PARAMS.capital


def test_late_final_bar_saves_realized_pnl_for_next_session(monitor):
    poll, record = monitor
    bars = pair(dip=70)
    for t in pd.date_range(at("15:20"), at("15:55"), freq="5min"):
        poll(t, bars)
    poll(at("17:31"), bars)
    finished = record()
    assert len(finished["trades"]) == 1 and finished["status"] == "closed"
    assert finished["state"]["pnl"] > 0
    poll(at("10:00", TOMORROW), pair(TOMORROW))
    assert record(TOMORROW)["opening_capital"] == pytest.approx(PARAMS.capital + finished["state"]["pnl"])


def test_feed_exception_sanitized_and_ledger_not_written(tmp_path):
    def failure():
        raise RuntimeError("private vendor request")

    path = tmp_path / "paper.json"
    with pytest.raises(paper._QuoteError) as exc:
        paper._poll_once(failure, PARAMS, path, lambda: at("11:01"))
    assert "private" not in str(exc.value) and not path.exists()
