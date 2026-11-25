from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from wquant import intraday_data
from wquant.scalp_data import ET, latest_pair, session_bounds, sessions


def bars(day, count=78):
    index = pd.date_range(f"{day} 09:30", periods=count, freq="5min", tz=ET)
    return pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0,
                         "volume": 100.0, "session": "RTH"}, index=index)


def test_half_day_and_holiday():
    holiday, half = date(2025, 11, 27), date(2025, 11, 28)
    assert session_bounds(holiday) is None
    pair = pd.concat([bars(holiday), bars(half, 42)])
    day, left, _, close = next(sessions(pair, pair))
    assert day == half and close.hour == 13 and len(left) == 42


def test_available_bars_require_contiguous_prefix_and_both_legs():
    full = bars("2025-11-26")
    cutoff = full.index[5] + pd.Timedelta(minutes=1)
    day, left, right, close = next(sessions(full, full, as_of=cutoff.tz_convert("UTC")))
    assert day == cutoff.date() and close.hour == 16
    assert left.index.equals(right.index) and left.index.equals(full.index[:5])
    assert not list(sessions(full, full.drop(full.index[4]), as_of=cutoff))
    assert not list(sessions(full, full.iloc[:3], as_of=cutoff))
    assert not list(sessions(full, full.iloc[:-1]))
    with pytest.raises(ValueError, match="时区"):
        list(sessions(full, full, as_of="2025-11-26 10:00"))


@pytest.mark.parametrize("column,value", [("low", 10.5), ("volume", 0), ("open", float("nan"))])
def test_ignores_bad_candles_and_duplicate_bars(column, value):
    full = bars("2025-11-26")
    bad = full.copy()
    bad.loc[bad.index[5], column] = value
    assert not list(sessions(full, bad))
    assert not list(sessions(full, pd.concat([full, full.iloc[:1]])))


def bar_row(timestamp, price=10):
    return dict(time=timestamp, open=price, high=price+1, low=price-1, close=price,
                volume=100, trading_session="RTH")


def client_for(response):
    api = Mock()
    api.get_batch_history_bar.return_value = response
    return SimpleNamespace(data=SimpleNamespace(market_data=api))


def test_latest_pair_by_symbol_with_each_webull_time_format():
    t = pd.Timestamp("2025-11-26 09:30", tz=ET)
    client = client_for({"result": [
        {"symbol": "SOXS", "result": [bar_row(str(int(t.timestamp()*1000)), price=20)]},
        {"symbol": "SOXL", "result": [bar_row(t.isoformat(), price=10)]},
    ]})
    left, right = latest_pair(client)
    assert left.index[0] == right.index[0] == t
    assert left.open.iloc[0] == 10 and right.open.iloc[0] == 20
    client.data.market_data.get_batch_history_bar.assert_called_once_with(
        ["SOXL", "SOXS"], "US_STOCK", "M5", count="100", trading_sessions="RTH",
        real_time_required=True,
    )
    with pytest.raises(ValueError, match="行情"):
        latest_pair(client_for([]))


def test_download_merges_only_raw_data(monkeypatch, tmp_path):
    monkeypatch.setattr(intraday_data, "DIR", tmp_path)
    old = intraday_data._to_frame([bar_row("2024-01-02T09:30:00-05:00", 10),
                                   bar_row("2024-01-03T09:30:00-05:00", 100)])
    old.to_csv(intraday_data.path("SOXS", "M5"))
    client = client_for([{"symbol": "SOXS", "result": [bar_row("2024-01-03T09:35:00-05:00", 101)]}])
    raw = intraday_data.download(client, "SOXS")
    assert list(raw.close) == [10, 100, 101]
    assert list(intraday_data.load("SOXS").close) == [100, 100, 101]


def test_download_never_caches_a_still_open_last_bar(monkeypatch, tmp_path):
    class Clock(datetime):
        @classmethod
        def now(cls, tz):
            return pd.Timestamp("2025-11-26 15:59", tz=ET).tz_convert(tz).to_pydatetime()

    monkeypatch.setattr(intraday_data, "datetime", Clock)
    monkeypatch.setattr(intraday_data, "DIR", tmp_path)
    rows = [bar_row("2025-11-26T15:50:00-05:00"), bar_row("2025-11-26T15:55:00-05:00")]
    intraday_data.download(client_for([{"symbol": "SOXL", "result": rows}]), "SOXL", days=0)
    cache = intraday_data.load("SOXL", adjust=False)
    assert list(cache.index.minute) == [50]
