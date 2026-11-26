from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

import wquant.futu_gateway as FG
import wquant.journal as journal
import wquant.paper_arena as PA
import wquant.swing_live as SL
from wquant.broker import LocalSimBroker
from wquant.config import load_settings

ET = SL.ET
DAY = date(2026, 11, 9)


def at(hhmm: str, day: date = DAY) -> datetime:
    return SL._at(day, hhmm)


def _bear_soxx(day: date) -> pd.DataFrame:
    """Create 120-bar falling SOXX daily DataFrame so 3-vote ensemble = -3."""
    dates = pd.bdate_range(end=day - timedelta(days=1), periods=120)
    close = pd.Series([200.0 - i * 0.8 for i in range(120)], index=dates)
    return pd.DataFrame({
        "open": close + 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1_000_000.0,
    })


class ArenaFakeMarket:
    def __init__(self, prices, prev, kospi=-0.02, macro_ok=False):
        self.prices = dict(prices)
        self.prev = dict(prev)
        self.kospi = kospi
        self.macro_ok = macro_ok
        self.daily_df = _bear_soxx(DAY)
        self.calls = 0

    def last_price(self, sym):
        self.calls += 1
        return datetime.now(ET), self.prices[sym]

    def minute_quote(self, sym, since, now):
        px = self.prices[sym]
        return now, px, px

    def prev_rth_close(self, sym, day):
        return self.prev[sym]

    def daily(self, sym, count):
        return self.daily_df

    def kospi_move(self, now):
        return self.kospi

    def path_stats(self, sym, day):
        px = self.prices[sym]
        return px, px, px

    def macro_short_allowed(self, day, gate_cfg):
        if not (gate_cfg or {}).get("enabled"):
            return True, ""
        return self.macro_ok, ("利率双紧缩 → 允许做空" if self.macro_ok else "利率未双紧缩 → 禁做空 SOXS")


def test_eval_macro_short_gate_no_lookahead_and_both_required():
    dates = [DAY - timedelta(days=i) for i in range(120, 0, -1)]
    # Add a same-day spike that MUST be ignored (< DAY only)
    irx_up = pd.Series([3.5] * 119 + [4.2], index=dates)
    irx_up[DAY] = 1.0  # future / same-day drop must not leak
    tnx_up = pd.Series([4.0] * 119 + [4.8], index=dates)
    tnx_dn = pd.Series([4.5] * 119 + [4.0], index=dates)

    cfg = {"enabled": True, "symbols": ["^IRX", "^TNX"], "sma_window": 60}
    ok, note = SL.eval_macro_short_gate(DAY, cfg, fetcher=lambda s: {"^IRX": irx_up, "^TNX": tnx_up}[s])
    assert ok is True and "允许夜盘做空 SOXS" in note

    ok2, note2 = SL.eval_macro_short_gate(DAY, cfg, fetcher=lambda s: {"^IRX": irx_up, "^TNX": tnx_dn}[s])
    assert ok2 is False and "禁夜盘做空 SOXS" in note2

    # Network error returns None (conservatively blocks this tick without locking the whole session)
    def boom(_):
        raise RuntimeError("timeout")
    ok3, note3 = SL.eval_macro_short_gate(DAY, cfg, fetcher=boom)
    assert ok3 is None and "下轮继续重试" in note3


def test_macro_gate_failure_does_not_lock_session_and_retries_next_tick(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "STATE", tmp_path / "state")
    monkeypatch.setattr(journal, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(SL, "KILL_SWITCH", tmp_path / "KILL_SWITCH")

    class FlakyMacroMarket(ArenaFakeMarket):
        def __init__(self):
            super().__init__({"SOXL": 96.0, "SOXS": 20.0}, {"SOXL": 100.0}, kospi=-0.02, macro_ok=True)
            self.attempts = 0

        def macro_short_allowed(self, day, gate_cfg):
            if not (gate_cfg or {}).get("enabled"):
                return True, ""
            self.attempts += 1
            if self.attempts == 1:
                return None, "利率门控拉取失败 → 本轮保守禁做空 SOXS，下轮继续重试"
            return True, "利率双紧缩 → 允许夜盘做空 SOXS"

    base = load_settings(environment="uat")
    md = FlakyMacroMarket()
    cached = SL.CachedMarket(md)
    runners = PA.build_arena_runners(base, ["plan_a"], a_mode="paper")

    # Tick 1 (21:00): fetch fails -> does NOT open SOXS, and leaves macro_short_allowed=None (not locked)
    s1 = PA.run_arena_tick(runners, cached, at("21:00"), day=DAY)["plan_a"]
    assert s1.qty == 0 and s1.macro_short_allowed is None and md.attempts == 1

    # Tick 2 (21:01): retries immediately (not cached) -> succeeds and opens SOXS
    s2 = PA.run_arena_tick(runners, cached, at("21:01"), day=DAY)["plan_a"]
    assert s2.macro_short_allowed is True and s2.qty > 0 and s2.symbol == "SOXS" and md.attempts == 2


def test_three_profiles_config_and_ledger_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "STATE", tmp_path / "state")
    monkeypatch.setattr(journal, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(SL, "KILL_SWITCH", tmp_path / "KILL_SWITCH")

    base = load_settings(environment="uat")
    pa = SL.swing_cfg(PA.load_profile_settings(base, PA.PROFILE_MAP["plan_a"]))["tactical"]
    pb = SL.swing_cfg(PA.load_profile_settings(base, PA.PROFILE_MAP["plan_b"]))["tactical"]
    po = SL.swing_cfg(PA.load_profile_settings(base, PA.PROFILE_MAP["plan_orig"]))["tactical"]

    assert pa["macro_short_gate"]["enabled"] is True and pa["reentry"] is True and pa["path"]["scan_to"] == "04:00"
    assert pb["macro_short_gate"]["enabled"] is True and pb["reentry"] is False and pb["path"]["scan_to"] == "23:59"
    assert po["macro_short_gate"]["enabled"] is False and po["reentry"] is True and po["path"]["scan_to"] == "04:00"

    # Loose-rate day: SOXL -4%, KOSPI -2% -> only plan_orig enters SOXS; plan_a and plan_b block SOXS
    md = ArenaFakeMarket({"SOXL": 96.0, "SOXS": 20.0}, {"SOXL": 100.0}, kospi=-0.02, macro_ok=False)
    runners = PA.build_arena_runners(base, ["plan_a", "plan_b", "plan_orig"],
                                     a_mode="paper", b_mode="paper", orig_mode="paper")
    shared = SL.CachedMarket(md)
    states = PA.run_arena_tick(runners, shared, at("21:00"), day=DAY)

    assert states["plan_a"].qty == 0 and states["plan_a"].macro_short_allowed is False
    assert states["plan_b"].qty == 0 and states["plan_b"].macro_short_allowed is False
    assert states["plan_orig"].qty > 0 and states["plan_orig"].symbol == "SOXS"

    # Ledgers stay completely separate
    assert SL.Books("uat").qty("tactical", "SOXS") == 0
    assert SL.Books("uat_plan_b").qty("tactical", "SOXS") == 0
    assert SL.Books("uat_plan_orig").qty("tactical", "SOXS") == states["plan_orig"].qty

    snap = {r["key"]: r for r in PA.arena_snapshot(base, prices={"SOXS": 22.0}, now=at("22:00"))}
    assert snap["plan_orig"]["unrealized"] > 0
    assert snap["plan_a"]["unrealized"] == 0.0 and snap["plan_b"]["unrealized"] == 0.0


def test_plan_b_blocks_late_night_and_reentry_while_plan_a_allows(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "STATE", tmp_path / "state")
    monkeypatch.setattr(journal, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(SL, "KILL_SWITCH", tmp_path / "KILL_SWITCH")

    base = load_settings(environment="uat")
    # 1) Late-night 01:30 signal during tight rates: plan_a enters, plan_b skips (scan_to=23:59)
    md = ArenaFakeMarket({"SOXL": 96.0, "SOXS": 20.0}, {"SOXL": 100.0}, kospi=-0.02, macro_ok=True)
    runners = PA.build_arena_runners(base, ["plan_a", "plan_b"], a_mode="paper", b_mode="paper")
    s1 = PA.run_arena_tick(runners, md, at("01:30"), day=DAY)
    assert s1["plan_a"].qty > 0 and s1["plan_a"].symbol == "SOXS"
    assert s1["plan_b"].qty == 0

    # 2) Same-day stop + 3% rebound on DAY2: plan_a re-enters leg 2, plan_b stays flat
    day2 = date(2026, 11, 10)
    md2 = ArenaFakeMarket({"SOXL": 96.0, "SOXS": 20.0}, {"SOXL": 100.0}, kospi=-0.02, macro_ok=True)
    md2.daily_df = _bear_soxx(day2)
    PA.run_arena_tick(runners, md2, at("21:00", day2), day=day2)
    md2.prices["SOXS"] = 18.7  # -6.5% -> stop out both A and B
    st_stop = PA.run_arena_tick(runners, md2, at("23:30", day2), day=day2)
    assert st_stop["plan_a"].stopped and st_stop["plan_b"].stopped
    md2.prices["SOXS"] = 19.4  # +3.7% rebound -> A re-enters, B does not
    st_reb = PA.run_arena_tick(runners, md2, at("10:00", day2), day=day2)
    assert st_reb["plan_a"].reentries == 1 and st_reb["plan_a"].qty > 0
    assert st_reb["plan_b"].reentries == 0 and st_reb["plan_b"].qty == 0


def test_futu_sim_broker_locks_simulate_and_falls_back_cleanly(monkeypatch):
    placed = []

    class FakeCtx:
        def get_acc_list(self):
            return 0, pd.DataFrame([
                {"acc_id": 1111, "trd_env": "REAL", "acc_status": "ACTIVE", "trdmarket_auth": ["US"]},
                {"acc_id": 2222, "trd_env": "SIMULATE", "acc_status": "ACTIVE", "trdmarket_auth": ["US"]},
            ])

        def place_order(self, **kwargs):
            placed.append(kwargs)
            if kwargs["session"] == "ALL":
                return -1, "simulate account does not support overnight"
            return 0, pd.DataFrame([{"order_id": "SIM-99"}])

        def close(self):
            pass

    fake_futu = SimpleNamespace(
        RET_OK=0,
        TrdMarket=SimpleNamespace(US="US"),
        SecurityFirm=SimpleNamespace(FUTUSECURITIES="FUTU"),
        TrdEnv=SimpleNamespace(SIMULATE="SIMULATE", REAL="REAL"),
        TrdSide=SimpleNamespace(BUY="BUY", SELL="SELL"),
        OrderType=SimpleNamespace(NORMAL="NORMAL"),
        TimeInForce=SimpleNamespace(DAY="DAY"),
        Session=SimpleNamespace(RTH="RTH", ETH="ETH", OVERNIGHT="OVERNIGHT", ALL="ALL"),
        OpenSecTradeContext=lambda **kw: FakeCtx(),
    )
    monkeypatch.setattr(FG, "_futu", lambda: fake_futu)
    broker = FG.FutuSimBroker({"type": "LIMIT", "limit_offset_bps": 30, "time_in_force": "DAY", "trading_session": "CORE"},
                              label="plan_b")
    assert broker.account["acc_id"] == 2222 and broker.account["trd_env"] == "SIMULATE"

    from wquant.risk import OrderIntent
    res_ok = broker.submit(OrderIntent("SOXS", "BUY", 10, 20.0, "test"))
    assert res_ok["status"] == "submitted" and placed[-1]["trd_env"] == "SIMULATE" and placed[-1]["acc_id"] == 2222

    broker_ovn = FG.FutuSimBroker({"type": "LIMIT", "limit_offset_bps": 30, "time_in_force": "DAY", "trading_session": "ALL_DAY"},
                                  label="plan_b", fallback_local=True)
    # Force session_for -> ALL_DAY so Futu session maps to "ALL" (rejected by mock)
    monkeypatch.setattr("wquant.broker.session_for", lambda cfg, now=None: "ALL_DAY")
    res_fb = broker_ovn.submit(OrderIntent("SOXS", "BUY", 10, 20.0, "ovn"))
    assert res_fb["status"] == "filled" and res_fb["fallback"] == "local"


def test_sanitize_futu_0400_spikes_removes_prev_rth_close_artifact():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-11-19 15:59:00", tz=ET),
        pd.Timestamp("2026-11-20 03:59:00", tz=ET),
        pd.Timestamp("2026-11-20 04:00:00", tz=ET),
        pd.Timestamp("2026-11-20 04:01:00", tz=ET),
    ])
    df = pd.DataFrame({
        "open": [63.10, 67.31, 63.14, 67.70],
        "high": [63.20, 67.40, 67.72, 68.10],
        "low": [63.05, 67.26, 63.14, 67.55],
        "close": [63.14, 67.34, 67.70, 67.99],
        "volume": [10000.0, 14000.0, 839000.0, 1770000.0],
        "session": ["RTH", "OVN", "PRE", "PRE"],
    }, index=idx)
    cleaned = FG.sanitize_futu_0400_spikes(df)
    bar_0400 = cleaned.loc[pd.Timestamp("2026-11-20 04:00:00", tz=ET)]
    assert bar_0400["open"] == pytest.approx(67.34)
    assert bar_0400["low"] == pytest.approx(67.34)
    assert bar_0400["high"] == pytest.approx(67.72)
    assert bar_0400["close"] == pytest.approx(67.70)


def test_futu_market_uses_live_1m_and_sanitizes_minute_quote(monkeypatch):
    k1_rows = pd.DataFrame([
        {"time_key": "2026-11-19 16:00:00", "open": 63.10, "high": 63.20, "low": 63.05, "close": 63.14, "volume": 10000},
        {"time_key": "2026-11-20 04:00:00", "open": 67.31, "high": 67.40, "low": 67.26, "close": 67.34, "volume": 14000},
        {"time_key": "2026-11-20 04:01:00", "open": 63.14, "high": 67.72, "low": 63.14, "close": 67.70, "volume": 839000},
        {"time_key": "2026-11-20 04:02:00", "open": 67.70, "high": 68.10, "low": 67.55, "close": 67.99, "volume": 1770000},
    ])

    class FakeQuoteCtx:
        def subscribe(self, codes, subtypes, **kw):
            return 0, None

        def get_cur_kline(self, code, num, ktype, autype):
            return 0, k1_rows

        def close(self):
            pass

    fake_futu = SimpleNamespace(
        RET_OK=0,
        SubType=SimpleNamespace(K_1M="K_1M", K_5M="K_5M"),
        KLType=SimpleNamespace(K_1M="K_1M", K_5M="K_5M", K_DAY="K_DAY"),
        AuType=SimpleNamespace(NONE="NONE", QFQ="QFQ"),
        Session=SimpleNamespace(ALL="ALL"),
        OpenQuoteContext=lambda **kw: FakeQuoteCtx(),
    )
    monkeypatch.setattr(FG, "_futu", lambda: fake_futu)
    now = datetime(2026, 11, 20, 4, 1, 30, tzinfo=ET)
    fm = FG.FutuMarket(lambda: now)
    fm._prev_rth_cache[("SOXS", date(2026, 11, 20))] = 63.14

    ts, px = fm.last_price("SOXS")
    assert px == pytest.approx(67.99)
    # minute_quote must only include closed bars (bar_end <= 04:01:30 -> excludes 04:01..04:02 unfinished bar)
    obs, close_px, hi = fm.minute_quote("SOXS", datetime(2026, 11, 20, 3, 59, tzinfo=ET), now)
    assert obs == datetime(2026, 11, 20, 4, 1, tzinfo=ET)
    assert close_px == pytest.approx(67.70)
    assert hi == pytest.approx(67.72)

