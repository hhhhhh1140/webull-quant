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


def test_view_gate_blocks_and_allows(env):
    s, books, tmp = env
    write_view(tmp, {"regime": "risk_off", "confidence": 0.8, "summary": "x"})
    md = FakeMarket({"SOXL": 104.0, "SOXS": 50.0}, {"SOXL": 100.0}, kospi=0.02)   # 做多信号
    br = FakeBroker()
    st = SL.tactical_step(s, md, br, books, SL.TacState(day=str(DAY)), at("21:00"))
    assert not br.orders and "只做 SOXS" in st.gate
    md2 = FakeMarket({"SOXL": 95.0, "SOXS": 50.0}, {"SOXL": 100.0}, kospi=-0.02)  # 做空信号
    st2 = SL.tactical_step(s, md2, br, books, SL.TacState(day=str(DAY + timedelta(days=1))), at("21:00", DAY + timedelta(days=1)))
    assert not br.orders or br.orders[-1][0] == "SOXS"


def test_view_gate_semis_override_and_low_confidence(env):
    s, books, tmp = env
    write_view(tmp, {"regime": "risk_off", "confidence": 0.9, "summary": "x",
                     "semis": {"bias": "none", "confidence": 0.8, "reason": "CPI 前观望"}})
    allowed, why = SL.view_gate(s, DAY)
    assert allowed == set() and "观望" in why
    write_view(tmp, {"regime": "risk_off", "confidence": 0.6, "summary": "x",
                     "semis": {"bias": "none", "confidence": 0.55, "reason": "方向不明"}})
    allowed, why = SL.view_gate(s, DAY)          # 低置信度的观望不拦
    assert allowed == {"long", "short"} and "不够" in why
    write_view(tmp, {"regime": "risk_off", "confidence": 0.5, "summary": "x"})
    allowed, _ = SL.view_gate(s, DAY)
    assert allowed == {"long", "short"}


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
