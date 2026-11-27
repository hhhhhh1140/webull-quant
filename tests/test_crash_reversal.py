"""Tests for Mode A (post-crash 1m right-side bottom reversal) across engine and all 3 paper profiles."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from wquant import journal, paper_arena as PA, swing_live as SL
from wquant.broker import LocalSimBroker
from wquant.config import Settings, load_settings

ET = SL.ET


def _make_settings(tmp_path, monkeypatch, *, enable_crash=True, dual=True, macro_gate=True):
    monkeypatch.setattr(journal, "STATE", tmp_path)
    monkeypatch.setattr(journal, "record", lambda *a, **kw: None)
    raw = {
        "swing": {
            "tactical": {
                "capital": 3000,
                "position_frac": 1.0,
                "stop_pct": 0.06,
                "trail_pct": 0.08,
                "reentry": True,
                "max_reentries": 1,
                "reentry_stop_pct": 0.06,
                "trend_switch": {"enabled": dual, "index": "SOXX", "core_symbol": "SOXL", "trail_pct": 0.12},
                "macro_short_gate": {"enabled": macro_gate, "symbols": ["^IRX", "^TNX"], "sma_window": 60},
                "path": {"enabled": True, "scan": "continuous", "scan_from": "21:00", "scan_to": "04:00", "veto_pct": 0.06},
                "sizing": {"mode": "fixed"},
                "crash_reversal": {
                    "enabled": enable_crash,
                    "drop_threshold": -0.25,
                    "start_delay_minutes": 0,
                    "entry_end": "09:45",
                    "wait_minutes": 20,
                    "bounce_pct": 0.02,
                    "ema_window": 20,
                    "position_frac": 1.0,
                    "stop_pct": 0.10,
                    "trail_arm_pct": 0.12,
                    "trail_pct": 0.10,
                },
            },
            "order": {"type": "LIMIT", "limit_offset_bps": 30, "time_in_force": "DAY", "trading_session": "ALL_DAY"},
        }
    }
    return Settings(raw=raw, environment="uat_test")


class FakeCrashMarket:
    def __init__(self, daily_map: dict[str, pd.DataFrame], bars_map: dict[str, pd.DataFrame], cur_prices: dict[str, float]):
        self.daily_map = daily_map
        self.bars_map = bars_map
        self.cur_prices = cur_prices
        self.minute_quote_val: tuple[datetime, float, float] | None = None

    def last_price(self, sym: str):
        return datetime(2026, 11, 23, 10, 0, tzinfo=ET), self.cur_prices[sym]

    def minute_bars(self, sym: str, since: datetime, now: datetime) -> pd.DataFrame:
        df = self.bars_map.get(sym, pd.DataFrame())
        if df.empty:
            return df
        return df.loc[(df.index >= since) & (df.index + timedelta(minutes=1) <= now)]

    def minute_quote(self, sym: str, since: datetime | None, now: datetime):
        return self.minute_quote_val

    def prev_rth_close(self, sym: str, day: date) -> float:
        df = self.daily_map[sym]
        return float(df.loc[df.index.date < day, "close"].iloc[-1])

    def daily(self, sym: str, count: int) -> pd.DataFrame:
        return self.daily_map[sym].tail(count)

    def kospi_move(self, now: datetime):
        return -0.02

    def path_stats(self, sym: str, day: date):
        px = self.cur_prices[sym]
        return px * 0.99, px * 1.01, px


def _build_1m_bars(start_et: datetime, closes: list[float], lows: list[float] | None = None, highs: list[float] | None = None) -> pd.DataFrame:
    idx = pd.DatetimeIndex([start_et + timedelta(minutes=i) for i in range(len(closes))], name="t")
    lows = lows or closes
    highs = highs or closes
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": [10_000.0] * len(closes)},
        index=idx,
    )


def test_all_three_paper_profiles_have_mode_a_enabled():
    base = load_settings(environment="uat")
    for prof in PA.PROFILES:
        cfg = SL.swing_cfg(PA.load_profile_settings(base, prof))
        cr = cfg["tactical"]["crash_reversal"]
        assert cr["enabled"] is True, f"{prof.key} should enable Mode A"
        assert cr["drop_threshold"] == pytest.approx(-0.25)
        assert cr["wait_minutes"] == 20
        assert cr["bounce_pct"] == pytest.approx(0.02)
        assert cr["stop_pct"] == pytest.approx(0.10)
        assert cr["trail_arm_pct"] == pytest.approx(0.12)
        assert cr["trail_pct"] == pytest.approx(0.10)


def test_eval_crash_reversal_entry_1m_requires_wait_bounce_and_ema():
    w_start = datetime(2026, 11, 20, 16, 0, tzinfo=ET)
    # 30 bars pre-close warmup at 31.0, then post-close low 30.0 at 16:00, flat/rising to 30.70 by 16:20
    closes = [31.0] * 30 + [30.0] + [30.2] * 19 + [30.75]
    lows = [30.9] * 30 + [30.0] + [30.1] * 19 + [30.60]
    bars = _build_1m_bars(w_start - timedelta(minutes=30), closes, lows=lows)
    cr_cfg = {"wait_minutes": 20, "bounce_pct": 0.02, "ema_window": 20}

    # At 16:15 (only 14m after 16:00 low): should NOT trigger yet
    now_early = datetime(2026, 11, 20, 16, 15, tzinfo=ET)
    ok_early, _, _, _ = SL.eval_crash_reversal_entry_1m(bars.loc[bars.index + timedelta(minutes=1) <= now_early], w_start, now_early, cr_cfg)
    assert ok_early is False

    # At 16:21 (20m after 16:00 low, close=30.75 >= 30.0*1.02=30.60, close > EMA20): MUST trigger!
    now_ok = datetime(2026, 11, 20, 16, 21, tzinfo=ET)
    ok, px, lo, desc = SL.eval_crash_reversal_entry_1m(bars.loc[bars.index + timedelta(minutes=1) <= now_ok], w_start, now_ok, cr_cfg)
    assert ok is True
    assert px == pytest.approx(30.75)
    assert lo == pytest.approx(30.0)
    assert "模式 A 1m" in desc


def test_mode_a_enters_in_afterhours_and_exits_on_trail_and_skips_normal_tactical(tmp_path, monkeypatch):
    # Friday 2026-11-20 SOXL crashes -31.8% (45.30 -> 30.88), next trading day is Monday 2026-11-23.
    settings = _make_settings(tmp_path, monkeypatch, enable_crash=True, dual=True, macro_gate=True)
    d_idx = pd.DatetimeIndex([date(2026, 11, 19), date(2026, 11, 20)], tz=ET)
    soxl_d = pd.DataFrame({"open": [48.0, 40.0], "high": [50.0, 42.0], "low": [44.0, 30.0], "close": [45.30, 30.88], "volume": [1e6, 2e6]}, index=d_idx)
    soxs_d = pd.DataFrame({"open": [60.0, 70.0], "high": [65.0, 85.0], "low": [58.0, 65.0], "close": [63.14, 83.16], "volume": [1e6, 2e6]}, index=d_idx)
    # SOXX downtrend (would normally block SOXL long!)
    soxx_idx = pd.bdate_range("2026-06-01", "2026-11-20", tz=ET)
    soxx_d = pd.DataFrame({"open": 200.0, "high": 201.0, "low": 199.0, "close": [250.0 - i * 0.5 for i in range(len(soxx_idx))], "volume": 1e6}, index=soxx_idx)

    w_start = datetime(2026, 11, 20, 16, 0, tzinfo=ET)
    closes = [31.2] * 30 + [30.5] + [30.7] * 19 + [31.50]
    lows = [31.0] * 30 + [30.5] + [30.6] * 19 + [31.30]
    bars_soxl = _build_1m_bars(w_start - timedelta(minutes=30), closes, lows=lows)

    md = FakeCrashMarket({"SOXL": soxl_d, "SOXS": soxs_d, "SOXX": soxx_d}, {"SOXL": bars_soxl}, {"SOXL": 31.50, "SOXS": 82.0})
    broker = LocalSimBroker(settings.raw["swing"]["order"], label="test")
    books = SL.Books(settings.environment)
    st = SL.TacState(day="2026-11-23")

    # 1) After-hours 2026-11-20 16:21 ET -> Mode A triggers & buys SOXL even though SOXX is downtrend!
    t_entry = datetime(2026, 11, 20, 16, 21, tzinfo=ET)
    st = SL.tactical_step(settings, md, broker, books, st, t_entry)
    assert st.crash_checked is True
    assert st.crash_symbol == "SOXL"
    assert st.crash_active is True
    assert st.decided is True
    assert st.qty > 0
    assert st.entry == pytest.approx(31.50)
    assert books.qty("tactical", "SOXL") == st.qty

    # 2) Monday 09:40 ET: rallies +20% to 38.00 -> arms trailing stop (+12% arm), holds
    t_peak = datetime(2026, 11, 23, 9, 40, tzinfo=ET)
    md.minute_quote_val = (t_peak, 38.00, 38.50)
    st = SL.tactical_step(settings, md, broker, books, st, t_peak)
    assert st.qty > 0
    assert st.peak == pytest.approx(38.50)
    assert st.done is False

    # 3) Monday 09:55 ET: pulls back 10% from 38.50 -> 34.60 <= 34.65 -> Mode A trailing take-profit exits & finishes day
    t_exit = datetime(2026, 11, 23, 9, 55, tzinfo=ET)
    md.minute_quote_val = (t_exit, 34.60, 35.00)
    st = SL.tactical_step(settings, md, broker, books, st, t_exit)
    assert st.qty == 0
    assert st.done is True
    assert st.pnl > 0
    assert books.qty("tactical", "SOXL") == 0


def test_mode_a_hard_stop_cuts_tariff_knife(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path, monkeypatch, enable_crash=True, dual=False, macro_gate=False)
    d_idx = pd.DatetimeIndex([date(2026, 11, 19), date(2026, 11, 20)], tz=ET)
    soxl_d = pd.DataFrame({"open": [48.0, 40.0], "high": [50.0, 42.0], "low": [44.0, 30.0], "close": [45.30, 30.88], "volume": [1e6, 2e6]}, index=d_idx)
    soxs_d = pd.DataFrame({"open": [60.0, 70.0], "high": [65.0, 85.0], "low": [58.0, 65.0], "close": [63.14, 83.16], "volume": [1e6, 2e6]}, index=d_idx)
    md = FakeCrashMarket({"SOXL": soxl_d, "SOXS": soxs_d}, {}, {"SOXL": 28.00, "SOXS": 90.0})
    broker = LocalSimBroker(settings.raw["swing"]["order"], label="test")
    books = SL.Books(settings.environment)
    books.add("tactical", "SOXL", 90)
    st = SL.TacState(
        day="2026-11-23", decided=True, side=1, symbol="SOXL", qty=90, entry=32.00, peak=32.00,
        trades=1, crash_checked=True, crash_symbol="SOXL", crash_active=True,
    )

    # Drops -10.5% to 28.64 <= 28.80 -> Mode A hard stop exits cleanly and marks day done (no reentry)
    t_sl = datetime(2026, 11, 23, 6, 30, tzinfo=ET)
    md.minute_quote_val = (t_sl, 28.64, 29.00)
    st = SL.tactical_step(settings, md, broker, books, st, t_sl)
    assert st.qty == 0
    assert st.done is True
    assert st.pnl < 0
    assert books.qty("tactical", "SOXL") == 0


def test_mode_a_does_not_fire_on_normal_drop(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path, monkeypatch, enable_crash=True, dual=False, macro_gate=False)
    d_idx = pd.DatetimeIndex([date(2026, 11, 19), date(2026, 11, 20)], tz=ET)
    # Only -18% drop (above -25% threshold) -> must NOT arm Mode A
    soxl_d = pd.DataFrame({"open": [48.0, 42.0], "high": [50.0, 43.0], "low": [44.0, 36.0], "close": [45.00, 36.90], "volume": [1e6, 2e6]}, index=d_idx)
    soxs_d = pd.DataFrame({"open": [60.0, 68.0], "high": [65.0, 76.0], "low": [58.0, 65.0], "close": [63.00, 74.34], "volume": [1e6, 2e6]}, index=d_idx)
    md = FakeCrashMarket({"SOXL": soxl_d, "SOXS": soxs_d}, {}, {"SOXL": 37.0, "SOXS": 74.0})
    broker = LocalSimBroker(settings.raw["swing"]["order"], label="test")
    books = SL.Books(settings.environment)
    st = SL.TacState(day="2026-11-23")

    st = SL.tactical_step(settings, md, broker, books, st, datetime(2026, 11, 20, 16, 15, tzinfo=ET))
    assert st.crash_checked is True
    assert st.crash_symbol == ""
    assert st.crash_active is False
    assert st.decided is False
    assert st.qty == 0
