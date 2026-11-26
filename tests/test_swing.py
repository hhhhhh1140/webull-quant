from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import wquant.journal as journal
import wquant.swing_live as SL
from wquant.config import Settings
from wquant.swing import SwingParams, backtest, ensemble_weight, target_series

ET = SL.ET
DAY = date(2026, 11, 9)


class FakeMarket:
    def __init__(self, prices, prev, kospi=0.0):
        self.prices, self.prev, self.daily_df, self.kospi = dict(prices), dict(prev), None, kospi

    def last_price(self, sym):
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

    path = None                       # (lo, hi, vwap)；None = 用当前价

    def path_stats(self, sym, day):
        if self.path is None:
            px = self.prices[sym]
            return px, px, px
        return self.path


class FakeBroker:
    def __init__(self):
        self.orders = []

    def submit(self, intent):
        self.orders.append((intent.symbol, intent.side, intent.quantity, intent.price))
        return {"status": "filled"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "STATE", tmp_path / "state")
    monkeypatch.setattr(journal, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(SL, "KILL_SWITCH", tmp_path / "KILL_SWITCH")
    import wquant.engine as engine
    monkeypatch.setattr(engine, "DECISIONS", tmp_path / "decisions")
    s = Settings(raw={"swing": {"tactical": {"poll_seconds": 60, "sizing": {"mode": "fixed"}}}})
    return s, SL.Books("uat"), tmp_path


def at(hhmm, day=DAY):
    return SL._at(day, hhmm)


def write_view(tmp_path, view, day=DAY):
    import json
    d = tmp_path / "decisions" / str(day)
    d.mkdir(parents=True, exist_ok=True)
    (d / "decisions.json").write_text(json.dumps({"date": str(day), "market_view": view, "decisions": []}))


def test_checkpoints_belong_to_previous_evening():
    assert at("21:00") == datetime(2026, 11, 8, 21, 0, tzinfo=ET)      # 周一交易日的 21:00 是周日晚上
    assert at("04:00") == datetime(2026, 11, 9, 4, 0, tzinfo=ET)


def test_waits_for_kospi_then_enters_at_later_checkpoint(env):
    s, books, _ = env
    md = FakeMarket({"SOXL": 104.0, "SOXS": 50.0}, {"SOXL": 100.0}, kospi=0.004)
    br = FakeBroker()
    st = SL.TacState(day=str(DAY))
    st = SL.tactical_step(s, md, br, books, st, at("20:59"))
    assert not st.evaluated and not br.orders
    st = SL.tactical_step(s, md, br, books, st, at("21:00"))         # SOXL +4%，但 KOSPI 只 +0.4%
    assert st.evaluated == ["21:00"] and not st.decided and not br.orders
    md.kospi = 0.015
    st = SL.tactical_step(s, md, br, books, st, at("23:00"))         # KOSPI 涨到 +1.5% → 进场
    assert st.decided and br.orders[-1][:3] == ("SOXL", "BUY", 9)     # 3000×33%/104 = 9 股


def test_full_day_stop_reentry_and_equity(env):
    s, books, _ = env
    md = FakeMarket({"SOXL": 104.0, "SOXS": 50.0}, {"SOXL": 100.0}, kospi=0.012)
    br = FakeBroker()
    st = SL.tactical_step(s, md, br, books, SL.TacState(day=str(DAY)), at("21:00"))
    assert books.qty("tactical", "SOXL") == 9
    for px, hh in ((100.0, "23:30"), (97.7, "02:00")):                 # 跌破 104×0.94 → 止损
        md.prices["SOXL"] = px
        st = SL.tactical_step(s, md, br, books, st, at(hh))
    assert st.stopped and books.qty("tactical", "SOXL") == 0
    for px, hh in ((95.0, "05:00"), (97.9, "06:00")):                  # 低点 95 反弹 3% 买回
        md.prices["SOXL"] = px
        st = SL.tactical_step(s, md, br, books, st, at(hh))
    assert st.reentries == 1 and br.orders[-1][:2] == ("SOXL", "BUY")
    md.prices["SOXL"] = 101.0
    st = SL.tactical_step(s, md, br, books, st, at("15:55"))
    assert st.done and books.qty("tactical", "SOXL") == 0
    expected = (97.7 - 104) * 9 + (101 - 97.9) * br.orders[2][2]
    assert SL.system_equity(s) == pytest.approx(3000 + expected, abs=0.01)


def test_no_signal_any_checkpoint_means_no_trade(env):
    s, books, _ = env
    md = FakeMarket({"SOXL": 101.0}, {"SOXL": 100.0}, kospi=0.02)
    br = FakeBroker()
    st = SL.TacState(day=str(DAY))
    for hh in ("21:00", "23:00", "01:30", "04:00"):
        st = SL.tactical_step(s, md, br, books, st, at(hh))
    assert st.done and not br.orders


def test_old_review_file_cannot_block_tactical_entry(env):
    s, books, tmp = env
    s.raw["swing"]["tactical"]["view_gate"] = True
    write_view(tmp, {"regime": "risk_off", "confidence": 0.9, "semis": {"bias": "none", "confidence": 0.9}})
    md = FakeMarket({"SOXL": 104.0}, {"SOXL": 100.0}, kospi=0.02)
    broker = FakeBroker()

    state = SL.tactical_step(s, md, broker, books, SL.TacState(day=str(DAY)), at("21:00"))

    assert state.qty > 0
    assert broker.orders[0][:2] == ("SOXL", "BUY")


def test_run_defaults_to_the_dual_layer_entry_point(monkeypatch):
    from wquant import cli

    settings = _dual_settings()
    monkeypatch.setattr(cli, "load_settings", lambda _: settings)
    calls = []
    monkeypatch.setattr(cli, "cmd_swing", lambda args, s: calls.append((args.what, args.mode, s)) or 0)

    assert cli.main(["run"]) == 0
    assert calls == [("tactical", "dry_run", settings)]


def test_korea_opposite_blocks_short(env):
    s, books, _ = env
    md = FakeMarket({"SOXL": 95.0, "SOXS": 40.0}, {"SOXL": 100.0}, kospi=0.012)   # SOXL −5%，KOSPI +1.2%
    br = FakeBroker()
    st = SL.TacState(day=str(DAY))
    for hh in ("21:00", "23:00", "01:30", "04:00"):
        st = SL.tactical_step(s, md, br, books, st, at(hh))
    assert st.done and not br.orders


def test_run_tactical_day_loop_terminates_and_resumes(env):
    s, books, _ = env
    md = FakeMarket({"SOXL": 104.0}, {"SOXL": 100.0}, kospi=0.02)
    br = FakeBroker()
    clock = {"t": at("20:00")}
    sleep = lambda sec: clock.__setitem__("t", clock["t"] + timedelta(seconds=sec))
    st = SL.run_tactical_day(s, md, br, books, lambda: clock["t"], sleep, day=DAY)
    assert st.done and [o[1] for o in br.orders] == ["BUY", "SELL"]
    SL.run_tactical_day(s, md, br, books, lambda: clock["t"], sleep, day=DAY)      # 重跑不重复下单
    assert len(br.orders) == 2


def test_core_rebalance_targets_weight(env):
    s, books, _ = env
    idx = pd.bdate_range("2025-01-01", periods=200)
    c = np.linspace(100, 200, 200)
    up = pd.DataFrame({"open": c, "high": c, "low": c * 0.99, "close": c}, index=idx)
    md = FakeMarket({"SOXL": 50.0}, {})
    md.daily_df = up
    br = FakeBroker()
    info = SL.core_rebalance(s, md, br, books)
    assert info["weight"] == 1.0 and info["target_qty"] == 600          # 10 万 × 30% / 50
    assert books.qty("core", "SOXL") == 600
    info = SL.core_rebalance(s, md, br, books)                         # 再跑一次：不重复下单
    assert len(br.orders) == 1


def test_trading_day_rolls_evening_to_next_session():
    assert SL.trading_day(datetime(2026, 11, 6, 20, 30, tzinfo=ET)) == date(2026, 11, 9)   # 周五晚 → 周一
    assert SL.trading_day(datetime(2026, 11, 9, 3, 0, tzinfo=ET)) == date(2026, 11, 9)


def test_swing_signals_no_lookahead():
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    df = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c}, index=idx)
    for p in (SwingParams(rule="donchian"), SwingParams(rule="ema"), SwingParams(rule="sma_mom")):
        full = target_series(df, p)
        part = target_series(df.iloc[:250], p)
        assert (full.iloc[:250] == part).all()


def test_position_frac_signal_only_default():
    t = SL.swing_cfg(Settings(raw={}))["tactical"]
    f, why = SL.position_frac(t, 0.06, 0.03, 0.05)          # 趋势效率关闭：0.05 不起作用
    assert f == pytest.approx(0.66) and "趋势效率" not in why


def test_position_frac_dynamic():
    t = SL.swing_cfg(Settings(raw={"swing": {"tactical": {"sizing": {"use_trend_efficiency": True}}}}))["tactical"]
    # 刚过门槛、震荡行情：33% × 1 × 0.6 = 20%
    f, why = SL.position_frac(t, 0.031, 0.011, 0.15)
    assert f == pytest.approx(0.33 * (0.031 / 0.03) * 0.6, rel=1e-3) and "趋势效率" in why
    # 强信号、单边趋势：33% × 2 × 1.3 = 85.8% → 上限 80%
    f, _ = SL.position_frac(t, 0.09, 0.03, 0.6)
    assert f == pytest.approx(0.80)
    # 信号倍数取 SOXL 和 KOSPI 的较小者：SOXL 6% → 2，KOSPI 1.2% → 1.2
    f, _ = SL.position_frac(t, -0.06, -0.012, 0.3)
    assert f == pytest.approx(0.33 * 1.2 * 1.0)
    # 固定模式
    t2 = {**t, "sizing": {"mode": "fixed"}}
    assert SL.position_frac(t2, 0.09, 0.03, 0.6)[0] == t["position_frac"]


def test_efficiency_ratio():
    trend = pd.Series(np.linspace(100, 70, 11))
    chop = pd.Series([100, 105, 100, 105, 100, 105, 100, 105, 100, 105, 101.0])
    assert SL.efficiency_ratio(trend, 10) == pytest.approx(1.0)
    assert SL.efficiency_ratio(chop, 10) == pytest.approx(1 / 49)
    assert SL.efficiency_ratio(trend.iloc[:5], 10) is None


def test_dynamic_entry_uses_trend_efficiency(env, monkeypatch):
    s, books, _ = env
    s.raw["swing"]["tactical"]["sizing"] = {"mode": "dynamic", "use_trend_efficiency": True}
    md = FakeMarket({"SOXL": 100.0, "SOXS": 50.0}, {"SOXL": 106.0}, kospi=-0.025)   # SOXL −5.7%，KOSPI −2.5%
    idx = pd.bdate_range("2026-10-01", periods=30)
    md.daily_df = pd.DataFrame({"close": np.linspace(200, 140, 30)}, index=idx)     # 单边下跌，效率 1.0
    br = FakeBroker()
    st = SL.tactical_step(s, md, br, books, SL.TacState(day=str(DAY)), at("21:00"))
    sig = min(0.0566 / 0.03, 2.5)            # SOXL 倍数 1.89，KOSPI 2.5，取小 → 封顶 2 以内
    assert st.frac == pytest.approx(min(0.33 * min(sig, 2.0) * 1.3, 0.80), rel=0.02)
    assert br.orders[0][0] == "SOXS" and br.orders[0][2] == int(3000 * st.frac // 50)
    assert "趋势效率" in st.sizing_note


def test_swing_cfg_deep_merge_keeps_defaults():
    cfg = SL.swing_cfg(Settings(raw={"swing": {"tactical": {"sizing": {"mode": "dynamic", "max": 0.5}}}}))
    sz = cfg["tactical"]["sizing"]
    assert sz["max"] == 0.5 and sz["er_tiers"] and sz["base"] == 0.33
    assert cfg["tactical"]["checkpoints"] == ["21:00", "23:00", "01:30", "04:00"]


def _run_path(s, books, md, br, path, sym="SOXL"):
    st = SL.TacState(day=str(DAY))
    for hh, px in path:
        if px is not None:
            md.prices[sym] = px
        st = SL.tactical_step(s, md, br, books, st, at(hh))
    return st


def test_pyramiding_only_outside_overnight_and_capped(env):
    s, books, _ = env
    md = FakeMarket({"SOXL": 100.0, "SOXS": 50.0}, {"SOXL": 96.0}, kospi=0.015)   # +4.2%，开 33%
    br = FakeBroker()
    st = _run_path(s, books, md, br, [("21:00", 100.0), ("03:00", 104.0)])        # 夜盘涨 4%：不能融资，不加仓
    assert st.adds == 0 and len(br.orders) == 1
    st2 = SL.tactical_step(s, md, br, books, st, at("04:30"))                     # 盘前：≥100×1.03 → 加仓
    assert st2.adds == 1
    for hh, px in (("05:00", 107.2), ("06:00", 110.5), ("07:00", 114.0), ("08:00", 118.0)):
        md.prices["SOXL"] = px
        st2 = SL.tactical_step(s, md, br, books, st2, at(hh))
    assert st2.frac_total <= 1.3 + 0.02 and st2.adds >= 3                        # 最多加到约 130%
    buys = [o for o in br.orders if o[1] == "BUY"]
    assert sum(o[2] for o in buys) == st2.qty == books.qty("tactical", "SOXL")


def test_no_add_after_cutoff(env):
    s, books, _ = env
    md = FakeMarket({"SOXL": 100.0}, {"SOXL": 96.0}, kospi=0.015)
    br = FakeBroker()
    st = _run_path(s, books, md, br, [("21:00", 100.0), ("15:10", 110.0)])
    assert st.adds == 0


def test_trailing_after_add_then_multiple_reentries(env):
    s, books, _ = env
    md = FakeMarket({"SOXL": 100.0}, {"SOXL": 96.0}, kospi=0.015)
    br = FakeBroker()
    path = [("21:00", 100.0), ("04:30", 104.0), ("05:00", 108.0),               # 加仓
            ("06:00", 101.0),                                                # 从 108 回撤 6.5% → 止盈
            ("06:30", 99.0), ("07:00", 102.5),                               # 低点 99 反弹 3.5% → 买回 1
            ("08:00", 95.0),                                                 # 跌破均价 6% → 止损
            ("09:00", 93.0), ("10:00", 96.0),                                # 低点 93 反弹 3.2% → 买回 2
            ("15:55", 97.0)]
    st = _run_path(s, books, md, br, path)
    reasons = " ".join(st.log)
    assert "回撤" in reasons and st.reentries == 2 and st.done
    assert books.qty("tactical", "SOXL") == 0
    sells = [o for o in br.orders if o[1] == "SELL"]
    assert len(sells) == 3                                                   # 止盈、止损、收盘平仓


# ---------------------------------------------------------------- 分时路径 / 连续扫描
PATH_ON = {"enabled": True, "vwap_filter": False, "veto_pct": 0.06, "scan": "continuous"}


def test_path_signal_disabled_is_plain_threshold():
    assert SL.path_signal(0.04, 104, 90, 110, 100, 0.03, {})[0] == 1
    assert SL.path_signal(0.01, 101, 90, 110, 100, 0.03, {})[0] == 0


def test_path_signal_veto_short_after_v_bounce():
    # 昨收 100，夜盘跌到 92（-8%），现在 96.5（-3.5%）：从低点已经反弹 4.9% 还没到 6%，可以空
    assert SL.path_signal(-0.035, 96.5, 92, 100, None, 0.03, PATH_ON)[0] == -1
    # 反弹到 97.6（-2.4%，从低点 +6.1%）本来就不够阈值；跌回 -3.1% 但从低点 +6.4%（低点 91）→ 否决
    side, why, _ = SL.path_signal(-0.031, 96.9, 91, 100, None, 0.03, PATH_ON)
    assert side == 0 and "不空" in why


def test_path_signal_vwap_filter_and_reversal():
    pc = {"enabled": True, "vwap_filter": True}
    assert SL.path_signal(0.04, 104, 95, 106, 105, 0.03, pc)[0] == 0          # 涨 4% 但在 VWAP 下方
    assert SL.path_signal(0.04, 104, 95, 106, 103, 0.03, pc)[0] == 1
    rev = {"enabled": True, "vwap_filter": True, "reversal_pct": 0.08}
    side, why, is_rev = SL.path_signal(0.01, 101, 92, 101, 98, 0.03, rev)     # -8% 拉回 +1%
    assert side == 1 and is_rev and "反转" in why


def test_continuous_scan_enters_between_checkpoints(tmp_path, monkeypatch, env):
    s, books, _ = env
    s = Settings(raw={"swing": {"tactical": {"poll_seconds": 60, "sizing": {"mode": "fixed"}, "path": PATH_ON}}})
    md = FakeMarket({"SOXL": 101.0, "SOXS": 50.0}, {"SOXL": 100.0}, kospi=0.012)
    br = FakeBroker()
    st = SL.TacState(day=str(DAY))
    st = SL.tactical_step(s, md, br, books, st, at("20:30"))          # 扫描窗口之前：什么都不做
    assert not st.log
    st = SL.tactical_step(s, md, br, books, st, at("21:00"))          # 检查时点：记一条
    n = len(st.log)
    for hh in ("21:01", "21:02", "21:30"):                              # 结论没变：不刷日志
        st = SL.tactical_step(s, md, br, books, st, at(hh))
    assert len(st.log) == n and not br.orders
    md.prices["SOXL"] = 103.5                                           # 22:15 涨过 3%，不用等 23:00
    st = SL.tactical_step(s, md, br, books, st, at("22:15"))
    assert st.decided and br.orders[-1][:2] == ("SOXL", "BUY")


def test_continuous_scan_vetoes_short_after_bounce(env):
    s, books, _ = env
    s = Settings(raw={"swing": {"tactical": {"poll_seconds": 60, "sizing": {"mode": "fixed"}, "path": PATH_ON}}})
    md = FakeMarket({"SOXL": 96.9, "SOXS": 50.0}, {"SOXL": 100.0}, kospi=-0.015)
    md.path = (91.0, 100.0, None)
    br = FakeBroker()
    st = SL.tactical_step(s, md, br, books, SL.TacState(day=str(DAY)), at("21:30"))
    assert not st.decided and not br.orders and "不空" in st.log[-1]


def test_continuous_scan_ends_after_window(env):
    s, books, _ = env
    s = Settings(raw={"swing": {"tactical": {"poll_seconds": 60, "sizing": {"mode": "fixed"}, "path": PATH_ON}}})
    md = FakeMarket({"SOXL": 100.5, "SOXS": 50.0}, {"SOXL": 100.0})
    br = FakeBroker()
    clock = {"t": at("20:50")}
    st = SL.run_tactical_day(s, md, br, books, lambda: clock["t"],
                             sleep_fn=lambda sec: clock.__setitem__("t", clock["t"] + timedelta(seconds=sec)))
    assert st.done and not br.orders and set(st.evaluated) == {"21:00", "23:00", "01:30", "04:00"}
    assert len(st.log) < 15                                             # 7 小时每分钟扫描，但日志不刷屏


def test_trailing_stop_from_peak_then_rebuy(env):
    s, books, _ = env
    s = Settings(raw={"swing": {"tactical": {"poll_seconds": 60, "sizing": {"mode": "fixed"}, "trail_pct": 0.07,
                                              "add_step": None}}})
    md = FakeMarket({"SOXL": 104.0, "SOXS": 50.0}, {"SOXL": 100.0}, kospi=0.012)
    br = FakeBroker()
    st = SL.tactical_step(s, md, br, books, SL.TacState(day=str(DAY)), at("21:00"))
    assert st.qty == 9
    for px, hh in ((120.0, "05:00"), (112.0, "10:00")):                 # 最高 120，回撤 6.7%：还不卖
        md.prices["SOXL"] = px
        st = SL.tactical_step(s, md, br, books, st, at(hh))
    assert st.qty == 9
    md.prices["SOXL"] = 111.0                                           # 120×0.93 = 111.6 → 卖（仍高于均价，不是普通止损）
    st = SL.tactical_step(s, md, br, books, st, at("10:30"))
    assert st.qty == 0 and st.stopped and "移动止损" in st.log[-1]
    for px, hh in ((108.0, "11:00"), (111.3, "11:30")):                 # 低点 108 反弹 3% 买回
        md.prices["SOXL"] = px
        st = SL.tactical_step(s, md, br, books, st, at(hh))
    assert st.reentries == 1 and br.orders[-1][:2] == ("SOXL", "BUY")


def _trend_df(up: bool, n: int = 120):
    idx = pd.bdate_range("2026-01-01", periods=n)
    c = np.linspace(100, 180, n) if up else np.linspace(180, 100, n)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c}, index=idx)


def _dual_settings():
    return Settings(raw={"swing": {"tactical": {
        "poll_seconds": 60, "position_frac": 1.0, "sizing": {"mode": "fixed"},
        "add_step": None, "trail_after_add": None, "trail_pct": 0.07,
        "stop_pct": 0.06, "reentry": True, "rebound_pct": 0.03,
        "max_reentries": 1, "reentry_stop_pct": 0.12,
        "trend_switch": {"enabled": True, "index": "SOXX", "core_symbol": "SOXL",
                         "core_frac": 1.0, "day_stop_pct": 0.12, "rebuy_pct": 0.05,
                         "core_session_start": "09:30"},
    }}})


def test_dual_uptrend_core_stop_and_next_day_rebuy(env):
    _, books, _ = env
    s = _dual_settings()
    md = FakeMarket({"SOXL": 100.0, "SOXS": 50.0}, {"SOXL": 100.0}, kospi=0.02)
    md.daily_df = _trend_df(up=True)
    br = FakeBroker()

    # 第 1 天：上涨趋势 → 09:30 买核心 SOXL，15:55 保留过夜不卖
    st1 = SL.TacState(day=str(DAY))
    st1 = SL.tactical_step(s, md, br, books, st1, at("21:00"))
    assert st1.trend_regime == 1 and st1.core_active_today and not br.orders
    st1 = SL.tactical_step(s, md, br, books, st1, at("09:30"))
    assert books.qty("core", "SOXL") == 30 and br.orders[-1][:2] == ("SOXL", "BUY")
    md.prices["SOXL"] = 140.0
    st1 = SL.tactical_step(s, md, br, books, st1, at("15:55"))
    assert not st1.done and books.qty("core", "SOXL") == 30
    assert SL._load_core_state("uat").peak == 140.0
    SL.tactical_step(s, md, br, books, st1, datetime(DAY.year, DAY.month, DAY.day, 20, tzinfo=ET))
    assert st1.done

    # The overnight session starts a new high. Yesterday's $140 cannot trigger today's stop.
    day2 = DAY + timedelta(days=1)
    st2 = SL.TacState(day=str(day2))
    md.prices["SOXL"] = 122.0
    st2 = SL.tactical_step(s, md, br, books, st2, at("21:00", day2))
    assert books.qty("core", "SOXL") == 30
    assert SL._load_core_state("uat").peak == 122.0
    md.prices["SOXL"] = 135.0
    SL.tactical_step(s, md, br, books, st2, at("22:00", day2))
    md.prices["SOXL"] = 118.0
    st2 = SL.tactical_step(s, md, br, books, st2, at("23:00", day2))
    assert st2.done and books.qty("core", "SOXL") == 0
    assert "核心移动止损" in st2.log[-1]
    cst = SL._load_core_state("uat")
    assert cst.stopped and cst.stopped_day == str(day2) and cst.peak == 0
    assert not cst.peak_checked_at
    md.prices["SOXL"] = 135.0
    SL.tactical_step(s, md, br, books, SL.TacState(day=str(day2)), at("10:00", day2))
    assert books.qty("core", "SOXL") == 0

    # 第 3 天：上一交易日收盘为 85，夜盘达到 89.25 买回。
    day3 = DAY + timedelta(days=2)
    st3 = SL.TacState(day=str(day3))
    md.prev["SOXL"] = 85.0
    md.prices["SOXL"] = 85.0
    st3 = SL.tactical_step(s, md, br, books, st3, at("20:01", day3))
    assert books.qty("core", "SOXL") == 0 and not st3.done
    md.prices["SOXL"] = 89.5
    st3 = SL.tactical_step(s, md, br, books, st3, at("20:02", day3))
    assert books.qty("core", "SOXL") > 0 and br.orders[-1][:2] == ("SOXL", "BUY")
    cst = SL._load_core_state("uat")
    assert not cst.stopped and cst.entry == 89.5 and cst.peak_day == ""


def test_core_waiting_to_rebuy_polls_before_first_tactical_checkpoint(env):
    _, books, _ = env
    s = _dual_settings()
    market = FakeMarket({"SOXL": 104.0}, {"SOXL": 100.0})
    market.daily_df = _trend_df(up=True)
    broker = FakeBroker()
    SL._save_core_state("uat", SL.CoreState(stopped=True, stopped_day=str(DAY - timedelta(days=3))))
    clock = {"time": at("20:01")}
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if broker.orders:
            clock["time"] = SL.market_window(DAY)[2]
        else:
            clock["time"] += timedelta(seconds=seconds)
            market.prices["SOXL"] = 106.0

    SL.run_tactical_day(s, market, broker, books, lambda: clock["time"], sleep_fn=sleep, day=DAY)
    assert sleeps[0] == 60
    assert broker.orders[0][:2] == ("SOXL", "BUY")


def test_regular_hours_buyback_waits_through_overnight(env):
    _, books, _ = env
    s = _dual_settings()
    s.raw["swing"]["tactical"]["trend_switch"]["rebuy_session"] = "RTH"
    market = FakeMarket({"SOXL": 106.0}, {"SOXL": 100.0})
    market.daily_df = _trend_df(up=True)
    broker = FakeBroker()
    SL._save_core_state("uat", SL.CoreState(stopped=True, stopped_day=str(DAY - timedelta(days=3))))
    state = SL.TacState(day=str(DAY))

    SL.tactical_step(s, market, broker, books, state, at("20:02"))
    assert not broker.orders
    SL.tactical_step(s, market, broker, books, state, at("09:31"))
    assert broker.orders[0][:2] == ("SOXL", "BUY")


def test_core_restarts_after_tactical_exit_and_stops_in_after_hours(env):
    _, books, _ = env
    s = _dual_settings()
    market = FakeMarket({"SOXL": 121.0}, {})
    broker = FakeBroker()
    books.add("core", "SOXL", 20)
    SL._save_core_state("uat", SL.CoreState(holding=True, entry=100.0, peak=140.0, peak_day=str(DAY),
                                           peak_checked_at=at("15:55").isoformat()))
    SL._save_state("uat", SL.TacState(day=str(DAY), done=True, decided=True, trend_votes=3, trend_regime=1))

    afternoon = datetime(DAY.year, DAY.month, DAY.day, 17, 1, tzinfo=ET)
    result = SL.run_tactical_day(s, market, broker, books, lambda: afternoon, day=DAY)

    assert result.done and broker.orders == [("SOXL", "SELL", 20, 121.0)]
    assert SL._load_core_state("uat").stopped_day == str(DAY)


def test_core_quote_reads_overnight_high_after_late_start(monkeypatch):
    start, _, close = SL.market_window(DAY)
    stamps = pd.date_range(start, periods=1401, freq="min")
    bars = pd.DataFrame({"close": 100., "high": 102.}, index=stamps)
    bars.loc[stamps[5], "high"] = 140.
    now = close - timedelta(minutes=30)
    md = SL.WebullMarket(None, lambda: now)

    def page(sym, span, count, end_time=None):
        available = bars.loc[:pd.Timestamp(end_time, unit="ms", tz="UTC")] if end_time else bars
        return available.tail(count)

    monkeypatch.setattr(md, "_bars", page)
    _, _, high = md.minute_quote("SOXL", start, now)
    assert high == 140.


def test_trading_window_includes_overnight_and_shortened_after_hours():
    thanksgiving = date(2026, 11, 26)
    half_day = date(2026, 11, 27)
    assert SL.market_window(thanksgiving) is None
    open_at, regular_close, closes = SL.market_window(half_day)
    assert open_at == datetime(2026, 11, 26, 20, tzinfo=ET)
    assert regular_close == datetime(2026, 11, 27, 13, tzinfo=ET)
    assert closes == datetime(2026, 11, 27, 17, tzinfo=ET)
    assert SL.trading_day(closes - timedelta(minutes=1)) == half_day
    next_day = date(2026, 11, 30)
    assert SL.next_market_window(closes)[:2] == (next_day, datetime(2026, 11, 29, 20, tzinfo=ET))


def test_core_quote_uses_completed_highs_after_cursor(monkeypatch):
    stamps = pd.date_range(at("09:29"), periods=4, freq="min")
    bars = pd.DataFrame({"close": [145., 126., 117., 110.], "high": [150., 140., 118., 170.]}, index=stamps)
    md = SL.WebullMarket(None, lambda: at("09:32"))
    monkeypatch.setattr(md, "_bars", lambda *args: bars)

    assert md.minute_quote("SOXL", at("09:30"), at("09:32")) == (at("09:32"), 117., 140.)
    assert md.minute_quote("SOXL", at("09:32"), at("09:32")) is None


def test_core_buyback_uses_previous_regular_close_each_trading_day(env):
    _, books, _ = env
    s = _dual_settings()
    market = FakeMarket({"SOXL": 82.0}, {"SOXL": 80.0})
    market.daily_df = _trend_df(up=True)
    broker = FakeBroker()
    SL._save_core_state("uat", SL.CoreState(stopped=True, stopped_day=str(DAY)))
    first_day = DAY + timedelta(days=1)
    second_day = DAY + timedelta(days=2)

    state = SL.tactical_step(s, market, broker, books, SL.TacState(day=str(first_day)), at("09:31", first_day))
    assert state.core_prev_close == 80.0 and not broker.orders
    market.prev["SOXL"] = 90.0
    market.prices["SOXL"] = 94.4
    state = SL.tactical_step(s, market, broker, books, SL.TacState(day=str(second_day)), at("09:31", second_day))
    assert state.core_prev_close == 90.0 and not broker.orders
    market.prices["SOXL"] = 94.6
    state.decided = True
    afternoon = datetime.combine(second_day, datetime.min.time(), ET) + timedelta(hours=17)
    SL.tactical_step(s, market, broker, books, state, afternoon)
    assert broker.orders[0][:2] == ("SOXL", "BUY")


def test_tactical_half_rebuy_trails_from_reentry_high(env):
    _, books, _ = env
    s = _dual_settings()
    s.raw["swing"]["tactical"].update(reentry_position_frac=.5, reentry_trail_pct=.12)
    market = FakeMarket({"SOXS": 50.0}, {})
    broker = FakeBroker()
    state = SL.TacState(day=str(DAY), decided=True, symbol="SOXS", side=-1, frac=1.0,
                        stopped=True, low_after_stop=48.0, trend_votes=-3, trend_regime=-1)
    SL.tactical_step(s, market, broker, books, state, at("08:00"))
    assert broker.orders == [("SOXS", "BUY", 30, 50.0)]

    market.minute_quote = lambda sym, since, now: (now, market.prices[sym], 60.0)
    market.prices["SOXS"] = 54.0
    SL.tactical_step(s, market, broker, books, state, at("08:05"))
    assert state.peak == 60.0 and state.qty == 30
    market.prices["SOXS"] = 52.7
    SL.tactical_step(s, market, broker, books, state, at("08:06"))
    assert broker.orders[-1] == ("SOXS", "SELL", 30, 52.7)
    assert state.qty == 0 and "第 2 腿移动止损" in state.log[-1]


def test_dual_downtrend_liquidates_core_blocks_long_and_uses_leg2_stop(env):
    _, books, _ = env
    s = _dual_settings()
    books.add("core", "SOXL", 20)
    SL._save_core_state("uat", SL.CoreState(holding=True, entry=100.0))

    # 趋势转跌：先清空核心 SOXL，再拦截做多信号，只允许做空 SOXS
    md = FakeMarket({"SOXL": 105.0, "SOXS": 50.0}, {"SOXL": 100.0}, kospi=0.02)
    md.daily_df = _trend_df(up=False)
    br = FakeBroker()
    st = SL.tactical_step(s, md, br, books, SL.TacState(day=str(DAY)), at("21:00"))
    assert st.trend_regime == -1 and books.qty("core", "SOXL") == 0
    assert br.orders[0][:2] == ("SOXL", "SELL") and len(br.orders) == 1  # 做多信号被三票下跌趋势拦截

    # 换成做空信号：全仓首腿触发 6% 止损，从低点反弹 3% 后半仓买回。
    s.raw["swing"]["tactical"]["reentry_position_frac"] = .50
    md2 = FakeMarket({"SOXL": 95.0, "SOXS": 50.0}, {"SOXL": 100.0}, kospi=-0.02)
    md2.daily_df = _trend_df(up=False)
    day2 = DAY + timedelta(days=1)
    br2 = FakeBroker()
    st2 = SL.tactical_step(s, md2, br2, books, SL.TacState(day=str(day2)), at("21:00", day2))
    assert br2.orders[-1][:2] == ("SOXS", "BUY")
    md2.prices["SOXS"] = 46.9                                           # -6.2% → 第 1 腿止损
    st2 = SL.tactical_step(s, md2, br2, books, st2, at("22:00", day2))
    assert st2.stopped and st2.qty == 0
    md2.prices["SOXS"] = 48.5                                           # 反弹 >3% → 第 2 腿买回
    st2 = SL.tactical_step(s, md2, br2, books, st2, at("23:00", day2))
    assert st2.reentries == 1 and st2.qty == 29
    assert br2.orders[-1] == ("SOXS", "BUY", 29, 48.5)
    md2.prices["SOXS"] = 44.5                                           # 较 48.5 跌 -8.2%（超 6% 但未到 12%）→ 不止损！
    st2 = SL.tactical_step(s, md2, br2, books, st2, at("10:00", day2))
    assert st2.qty > 0
    md2.prices["SOXS"] = 42.5                                           # 跌 -12.4% → 触发第 2 腿 12% 兜底止损，且不再买回
    st2 = SL.tactical_step(s, md2, br2, books, st2, at("11:00", day2))
    assert st2.qty == 0 and "第 2 腿兜底止损" in st2.log[-1]
    md2.prices["SOXS"] = 46.0
    st2 = SL.tactical_step(s, md2, br2, books, st2, at("12:00", day2))
    assert st2.qty == 0 and st2.reentries == 1                          # max_reentries=1，不再开第 3 腿
