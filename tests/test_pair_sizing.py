from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wquant.pair_sizing import BASE, allocation, load_signals, replay_sizing
from wquant.pair_study import CAPITAL, simulate
from wquant.sizing_news import NewsBook, NewsDecision


def sample(tmp_path):
    idx = pd.DatetimeIndex(["2026-06-17 03:55","2026-06-17 04:00","2026-06-17 04:05",
                            "2026-06-17 09:30","2026-06-17 15:55"], tz="America/New_York")
    f = pd.DataFrame({"session": ["OVN","PRE","PRE","RTH","RTH"], "open": [106,100,100,100,102],
                      "high": [107,101,101,101,103], "low": [104,98,98,98,99],
                      "close": [106,99,100,99,101], "volume": 500}, index=idx)
    d = SimpleNamespace(L=f, S=f.copy(), date=idx[-1].date(), night_signal_close=106.3, prev_close=100,
                        korea_at_four=.011)
    path = tmp_path / "earnings.csv"
    path.write_text("issuer,published_at,source\nAMD,2026-06-17T04:01:00-04:00,fixture\n")
    return load_signals([d], NewsBook(), path)[0]


def test_clock_and_news_gating(tmp_path):
    s = sample(tmp_path)
    assert s.strong and s.korea == .011 and s.earnings == ()
    assert allocation(s, "events") == 1 and allocation(s, "news") == BASE
    down = replace(s, order=replace(s.order, side=-1), move=-.07, korea=-.015)
    yes = replace(down, news=NewsDecision("directional", -1, False, 1))
    assert allocation(yes, "news") == 1
    assert allocation(replace(yes, news=replace(yes.news, scheduled_risk=True)), "news") == BASE
    assert allocation(replace(yes, earnings=("AMD",)), "events") == BASE
    assert allocation(replace(yes, korea=np.nan), "strong") == BASE
    assert allocation(replace(yes, news=NewsDecision("conflict", 0, False, 2)), "news") == 0
    assert allocation(replace(yes, news=NewsDecision("directional", 1, False, 1)), "news") == 0


def test_full_cash_no_borrow_fees_and_stress(tmp_path):
    s = sample(tmp_path)
    full, = simulate(s.day, [s.order], CAPITAL, 1, notional=CAPITAL)
    assert full.qty == 29 and full.qty * full.entry_price + .20 <= CAPITAL
    records, dd = replay_sizing([s,s], "full")
    assert sum(r["full"] for r in records) == 2
    assert records[0]["pnl"] == pytest.approx(full.pnl)
    assert records[1]["return"] == pytest.approx(records[1]["pnl"] / (CAPITAL + full.pnl))
    assert dd < 0  # profitable exit still had a marked dip
    conflict = replace(s, news=NewsDecision("conflict", 0, False, 2))
    none, dd = replay_sizing([conflict], "news")
    assert none[0]["filled"] is False and dd == 0
    delayed, _ = replay_sizing([s], "full", delay=1, cost=2)
    assert delayed[0]["pnl"] < full.pnl
