import pandas as pd
import pytest

from wquant.backtest import run_backtest
from wquant.broker import build_webull_order
from wquant.data import synthetic_bars
from wquant.indicators import add_indicators, rsi
from wquant.risk import OrderIntent, RiskManager
from wquant.strategy import generate_signals
from wquant.webull_client import bars_to_frame

RISK = {"max_position_pct": 0.10, "max_positions": 3, "max_order_notional": 5000,
        "max_daily_loss_pct": 0.03, "cash_buffer_pct": 0.05, "min_order_notional": 100}
STRAT = {"sma_fast": 20, "sma_slow": 50, "rsi_period": 14, "rsi_max_entry": 70,
         "momentum_lookback": 20, "atr_period": 14, "atr_stop_mult": 2.5}


def test_rsi_bounds():
    s = pd.Series(range(1, 60), dtype=float)
    r = rsi(s).dropna()
    assert (r >= 0).all() and (r <= 100).all()
    assert r.iloc[-1] == 100  # 单边上涨


def test_indicators_no_lookahead():
    df = synthetic_bars("X", 300, seed=1)
    full = add_indicators(df, STRAT)
    part = add_indicators(df.iloc[:200], STRAT)
    pd.testing.assert_frame_equal(full.iloc[:200], part)


def test_risk_caps_and_whitelist():
    rm = RiskManager(RISK, ["AAPL"])
    d = rm.size_buy("AAPL", 100, equity=100_000, cash=100_000, n_positions=0)
    assert d.approved and d.intent.quantity == 50          # min(10000, 5000) / 100
    assert not rm.size_buy("TSLA", 100, 100_000, 100_000, 0).approved
    assert not rm.size_buy("AAPL", 100, 100_000, 100_000, 3).approved
    # 当日亏损 4% > 3% 熔断
    assert not rm.size_buy("AAPL", 100, 96_000, 96_000, 0, start_of_day_equity=100_000).approved
    # conviction 只能调低仓位
    half = rm.size_buy("AAPL", 100, 40_000, 40_000, 0, conviction=0.5)
    assert half.intent.quantity == 20


def test_kill_switch(tmp_path, monkeypatch):
    import wquant.risk as risk
    ks = tmp_path / "KILL_SWITCH"
    ks.touch()
    monkeypatch.setattr(risk, "KILL_SWITCH", ks)
    assert not RiskManager(RISK, ["AAPL"]).size_buy("AAPL", 100, 1e5, 1e5, 0).approved


def test_order_format():
    o = build_webull_order(OrderIntent("AAPL", "BUY", 10, 200.0), {"type": "LIMIT", "limit_offset_bps": 20})
    assert o["quantity"] == "10" and o["limit_price"] == "200.40" and o["side"] == "BUY"
    assert o["instrument_type"] == "EQUITY" and o["support_trading_session"] == "CORE"
    m = build_webull_order(OrderIntent("AAPL", "SELL", 5, 200.0), {"type": "MARKET"})
    assert "limit_price" not in m


def test_bars_to_frame_ms_and_iso():
    rows = [{"time": 1730851200000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}]
    assert bars_to_frame(rows)["close"].iloc[0] == 1.5
    rows = [{"time": "2026-11-05T05:00:00.000+0000", "open": "1", "high": "2", "low": "0.5", "close": "1.5", "volume": "10"}]
    assert str(bars_to_frame(rows).index[0].date()) == "2026-11-05"


def test_signals_exit_on_stop():
    df = add_indicators(synthetic_bars("X", 300, seed=2), STRAT)
    sigs = generate_signals({"X": df}, {"X": df["close"].iloc[-1] * 10}, STRAT)  # 极高的跟踪高点 → 必触发止损
    assert sigs and sigs[0].action == "SELL"


def test_backtest_runs_and_respects_limits():
    bars = {s: synthetic_bars(s, 600, seed=i) for i, s in enumerate(["A", "B", "C", "D", "E"])}
    res = run_backtest(bars, STRAT, RISK, {"initial_cash": 100_000, "slippage_bps": 5})
    assert len(res.equity) > 0 and res.metrics["trades"] >= 0
    assert all(t.qty * t.entry_price <= 5000 * 1.01 for t in res.trades)
    # 任一时刻持仓数不超过上限
    for d in res.equity.index[::20]:
        open_ = [t for t in res.trades if t.entry_date <= d and (t.exit_date is None or t.exit_date > d)]
        assert len(open_) <= 3


def _isolate(tmp_path, monkeypatch):
    import wquant.broker as broker
    import wquant.engine as engine
    import wquant.journal as journal
    monkeypatch.setattr(journal, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(journal, "STATE", tmp_path / "state")
    monkeypatch.setattr(broker, "ROOT", tmp_path)
    monkeypatch.setattr(engine, "DECISIONS", tmp_path / "decisions")
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    return engine


def _settings():
    from wquant.config import Settings
    return Settings(raw={"universe": ["QQQ", "NVDA", "GOOGL", "MSFT"], "mode": "paper", "strategy": STRAT,
                         "risk": RISK, "ai": {"mode": "claude_code", "min_confidence": 0.6},
                         "order": {"type": "LIMIT"}, "backtest": {"initial_cash": 100000}})


def test_scan_then_execute_with_claude_code_decisions(tmp_path, monkeypatch):
    import json
    from datetime import date
    engine = _isolate(tmp_path, monkeypatch)
    s = _settings()
    path = engine.scan(s, source="synthetic")
    packet = json.loads(path.read_text())
    buys = [c["symbol"] for c in packet["candidates"] if c["action"] == "BUY"]
    assert len(buys) >= 2

    # 没有 decisions.json 时拒绝执行
    with pytest.raises(FileNotFoundError):
        engine.execute(s, review="file", source="synthetic")

    # 批准第一个，否决第二个，第三个（若有）漏写 → 视为否决
    decs = [{"symbol": buys[0], "action": "APPROVE", "confidence": 0.8, "conviction": 0.5, "reasoning": "ok", "key_risks": []},
            {"symbol": buys[1], "action": "REJECT", "confidence": 0.8, "conviction": 0, "reasoning": "财报临近", "key_risks": ["财报"]}]
    (path.parent / "decisions.json").write_text(json.dumps({"date": str(date.today()), "decisions": decs}))
    res = {r["symbol"]: r["outcome"] for r in engine.execute(s, review="file", source="synthetic")}
    assert res[buys[0]] == "filled"
    assert res[buys[1]] == "rejected_by_review"
    for sym in buys[2:]:
        assert res[sym] == "rejected_by_review"


def test_stale_decisions_rejected(tmp_path, monkeypatch):
    import json
    engine = _isolate(tmp_path, monkeypatch)
    s = _settings()
    path = engine.scan(s, source="synthetic")
    (path.parent / "decisions.json").write_text(json.dumps({"date": "2020-01-01", "decisions": []}))
    with pytest.raises(ValueError):
        engine.execute(s, review="file", source="synthetic")


def test_no_api_import_in_default_mode():
    import subprocess, sys
    code = "import sys, wquant.engine, wquant.cli; print('anthropic' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_own_position_ledger_ignores_foreign_holdings(tmp_path, monkeypatch):
    engine = _isolate(tmp_path, monkeypatch)
    import wquant.journal as journal
    journal.save_owned("uat", {"NVDA": 10, "QQQ": 5})
    actual = {"NVDA": {"quantity": 300, "cost_price": 100, "last_price": 120},   # 别人另外买了 290 股
              "AAPL": {"quantity": 50, "cost_price": 200, "last_price": 210}}    # 别人的持仓
    mine = engine._own_positions("uat", actual, busy={"QQQ"})
    assert mine == {"NVDA": {"quantity": 10, "cost_price": 100, "last_price": 120}}
    assert journal.owned("uat") == {"NVDA": 10, "QQQ": 5}      # QQQ 订单未成交，保留
    engine._own_positions("uat", actual, busy=set())
    assert journal.owned("uat") == {"NVDA": 10}                # QQQ 订单已失效，移除
    engine._update_ledger("uat", "SELL", "NVDA", 10)
    assert journal.owned("uat") == {}


def test_sandbox_and_live_keys_are_separate(monkeypatch):
    from wquant.config import load_settings
    monkeypatch.setenv("WEBULL_APP_KEY", "livekey123")
    monkeypatch.setenv("WEBULL_APP_SECRET", "livesecret")
    monkeypatch.setenv("WEBULL_SANDBOX_APP_KEY", "sandkey456")
    monkeypatch.setenv("WEBULL_SANDBOX_APP_SECRET", "sandsecret")
    monkeypatch.setenv("WEBULL_ENVIRONMENT", "uat")
    s = load_settings()
    assert s.app_key == "sandkey456" and s.token_dir.endswith("uat-sandkey4")
    monkeypatch.setenv("WEBULL_ENVIRONMENT", "prod")
    s = load_settings()
    assert s.app_key == "livekey123" and s.token_dir.endswith("prod-livekey1")


def test_extended_session_forces_limit():
    o = build_webull_order(OrderIntent("AAPL", "BUY", 1, 100.0), {"type": "MARKET", "trading_session": "ALL_DAY"})
    assert o["order_type"] == "LIMIT" and o["support_trading_session"] == "ALL_DAY" and o["limit_price"] == "100.20"


def test_cache_merge_keeps_history(tmp_path, monkeypatch):
    import wquant.data as data
    monkeypatch.setattr(data, "CACHE", tmp_path)
    full = synthetic_bars("X", 300, seed=3)
    data.save_cache("X", full)
    data.save_cache("X", full.tail(50) * 1.01)      # 只拉了最近 50 根
    merged = data.load_csv(tmp_path / "X.csv")
    assert len(merged) == 300
    assert merged["close"].iloc[-1] == pytest.approx(full["close"].iloc[-1] * 1.01)


def test_market_filter_blocks_buys_in_scan_and_backtest(tmp_path, monkeypatch):
    import json
    from wquant.strategy import market_regime
    engine = _isolate(tmp_path, monkeypatch)
    # 指数一路下跌 → 跌破 SMA200
    down = synthetic_bars("IDX", 400, seed=5)
    down["close"] = down["close"].iloc[0] * (0.998 ** pd.Series(range(400), index=down.index))
    for c in ("open", "high", "low"):
        down[c] = down["close"]
    strat = {**STRAT, "market_filter": {"enabled": True, "symbols": ["IDX"], "ma": "sma_slow", "mode": "any"}}
    reg = market_regime({"IDX": add_indicators(down, strat)}, strat)
    assert reg["risk_off"] and reg["weak"] == ["IDX"]

    import wquant.data as data
    orig = data.synthetic_bars
    monkeypatch.setattr(engine, "load_bars", lambda st, syms, source, count: {
        s: (down if s == "IDX" else orig(s, count)) for s in syms})
    s = _settings()
    s.raw["strategy"] = strat
    path = engine.scan(s, source="synthetic")
    packet = json.loads(path.read_text())
    assert packet["market"]["risk_off"]
    assert all(c["action"] != "BUY" for c in packet["candidates"])
    assert packet["blocked_buys"]

    bars = {sym: synthetic_bars(sym, 400, seed=i) for i, sym in enumerate(["A", "B", "C"])}
    bars["IDX"] = down
    res = run_backtest(bars, strat, RISK, {"initial_cash": 100_000}, universe=["A", "B", "C"])
    late = [t for t in res.trades if t.entry_date > down.index[250]]
    assert not late          # SMA200 有效且指数在其下方后，不再开新仓
    assert all(t.symbol != "IDX" for t in res.trades)


def test_leverage_and_group_cap():
    rm = RiskManager({**RISK, "max_per_group": 2, "max_positions": 10}, ["SOXL", "MU", "WDC"])
    d = rm.size_buy("SOXL", 40, 100_000, 100_000, 0, leverage=3, group="semis_3x")
    assert d.approved and d.intent.notional <= 5000 / 3 + 40     # 单笔上限也按 1/3
    assert not rm.size_buy("WDC", 100, 100_000, 100_000, 2, group="memory", group_count=2).approved
    assert rm.size_buy("WDC", 100, 100_000, 100_000, 2, group="memory", group_count=1).approved


def test_settings_groups_and_instruments():
    from wquant.config import Settings
    s = Settings(raw={"universe": {"memory": ["mu", "WDC"], "semis_3x_short": ["SOXS"]},
                      "instruments": {"SOXS": {"leverage": 3, "inverse": True}}})
    assert s.universe == ["MU", "WDC", "SOXS"] and s.groups["MU"] == "memory"
    assert s.instrument("soxs") == {"leverage": 3.0, "inverse": True}
    assert s.instrument("MU") == {"leverage": 1.0, "inverse": False}
    assert Settings(raw={"universe": ["aapl"]}).groups == {"AAPL": "default"}


def test_inverse_etf_exempt_from_market_filter_in_backtest():
    down = synthetic_bars("IDX", 400, seed=5)
    down["close"] = down["close"].iloc[0] * (0.998 ** pd.Series(range(400), index=down.index))
    for c in ("open", "high", "low"):
        down[c] = down["close"]
    up = synthetic_bars("INV", 400, seed=7)   # 指数跌，反向 ETF 带噪声地上涨
    trend = 1.002 ** pd.Series(range(400), index=up.index)
    for c in ("open", "high", "low", "close"):
        up[c] = up[c] / up["close"].iloc[0] * 50 * trend
    strat = {**STRAT, "market_filter": {"enabled": True, "symbols": ["IDX"], "ma": "sma_slow", "mode": "any"}}
    bars = {"IDX": down, "INV": up}
    res = run_backtest(bars, strat, RISK, {"initial_cash": 100_000}, universe=["INV"],
                       instruments={"INV": {"leverage": 3, "inverse": True}})
    late = [t for t in res.trades if t.entry_date > down.index[250]]
    assert late, "大盘弱势时反向 ETF 应该仍可买入"
    assert all(t.qty * t.entry_price <= 5000 / 3 * 1.01 for t in res.trades)


INV_STRAT = {**STRAT, "inverse_rule": {"sma_fast": 20, "sma_slow": 50, "momentum_lookback": 10,
                                       "rsi_max_entry": 80, "atr_stop_mult": 2.0, "max_hold_days": 15,
                                       "require_weak_index": True}}


def _uptrend(n=300, seed=7):
    df = synthetic_bars("INV", n, seed=seed)
    trend = 1.003 ** pd.Series(range(n), index=df.index)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] / df["close"].iloc[0] * 50 * trend
    return add_indicators(df, INV_STRAT)


def test_inverse_entry_needs_weak_index():
    from wquant.strategy import inverse_entry_ok
    df = _uptrend()
    row = next(r for _, r in df.iloc[::-1].iterrows() if r.close > r.st_fast > r.st_slow and r.st_mom > 0 and r.rsi < 80)
    assert inverse_entry_ok(row, INV_STRAT, index_weak=True)
    assert not inverse_entry_ok(row, INV_STRAT, index_weak=False)


def test_inverse_exit_on_hold_limit():
    from wquant.strategy import generate_signals
    df = _uptrend()
    last = df.index[-1]
    entry = (last - pd.tseries.offsets.BDay(20)).date()
    sigs = generate_signals({"INV": df}, {"INV": float(df["close"].iloc[-1])}, INV_STRAT,
                            inverse={"INV"}, entry_dates={"INV": str(entry)})
    assert sigs and sigs[0].action == "SELL"
    recent = (last - pd.tseries.offsets.BDay(3)).date()
    sigs2 = generate_signals({"INV": df}, {"INV": float(df["close"].iloc[-1])}, INV_STRAT,
                             inverse={"INV"}, entry_dates={"INV": str(recent)})
    assert not sigs2 or "持有满" not in sigs2[0].reason


def test_inverse_backtest_respects_hold_limit():
    down = synthetic_bars("IDX", 400, seed=5)
    down["close"] = down["close"].iloc[0] * (0.998 ** pd.Series(range(400), index=down.index))
    for c in ("open", "high", "low"):
        down[c] = down["close"]
    strat = {**INV_STRAT, "market_filter": {"enabled": True, "symbols": ["IDX"], "ma": "sma_slow", "mode": "any"}}
    inv = synthetic_bars("INV", 400, seed=7)
    trend = 1.003 ** pd.Series(range(400), index=inv.index)
    for c in ("open", "high", "low", "close"):
        inv[c] = inv[c] / inv["close"].iloc[0] * 50 * trend
    res = run_backtest({"IDX": down, "INV": inv}, strat, RISK, {"initial_cash": 100_000}, universe=["INV"],
                       instruments={"INV": {"leverage": 3, "inverse": True}})
    assert res.trades
    for t in res.trades:
        if t.reason != "回测结束":
            held = ((inv.index > t.entry_date) & (inv.index < t.exit_date)).sum()
            assert held <= 15


def test_inverse_rule_can_be_disabled():
    from wquant.strategy import inverse_entry_ok
    df = _uptrend()
    row = next(r for _, r in df.iloc[::-1].iterrows() if r.close > r.st_fast > r.st_slow and r.st_mom > 0 and r.rsi < 80)
    off = {**INV_STRAT, "inverse_rule": {**INV_STRAT["inverse_rule"], "enabled": False}}
    assert not inverse_entry_ok(row, off, index_weak=True)


def _hedge_settings():
    from wquant.config import Settings
    return Settings(raw={
        "universe": {"megacap": ["QQQ", "NVDA", "GOOGL", "MSFT"], "semis_3x_short": ["SOXS"]},
        "instruments": {"SOXS": {"leverage": 3, "inverse": True}},
        "mode": "paper", "strategy": STRAT, "risk": {**RISK, "max_positions": 8},
        "ai": {"mode": "claude_code", "min_confidence": 0.6}, "order": {"type": "LIMIT"},
        "backtest": {"initial_cash": 100000},
        "hedge": {"enabled": True, "instruments": ["SOXS"], "min_confidence": 0.7, "max_position_pct": 0.03,
                  "max_hold_days": 10, "atr_stop_mult": 2.0, "exit_when_view_changes": True}})


def _write_review(path, view, hedges, decisions=()):
    import json
    from datetime import date
    (path.parent / "decisions.json").write_text(json.dumps(
        {"date": str(date.today()), "market_view": view, "hedges": hedges, "decisions": list(decisions)}))


def test_hedge_opens_only_with_confident_risk_off(tmp_path, monkeypatch):
    import json
    engine = _isolate(tmp_path, monkeypatch)
    import wquant.journal as journal
    s = _hedge_settings()
    path = engine.scan(s, source="synthetic")
    packet = json.loads(path.read_text())
    assert "SOXS" in packet["hedge_quotes"] and "breadth" in packet["sentiment"]
    buys = [c["symbol"] for c in packet["candidates"] if c["action"] == "BUY" and c["symbol"] != "SOXS"]

    # 中性研判：对冲被拒
    _write_review(path, {"regime": "neutral", "confidence": 0.9, "summary": "x"},
                  [{"symbol": "SOXS", "conviction": 1.0, "reasoning": "test"}])
    res = engine.execute(s, review="file", source="synthetic")
    assert [r for r in res if r["signal"] == "HEDGE"][0]["outcome"] == "hedge_rejected"

    # 偏空但置信度不够：拒绝
    _write_review(path, {"regime": "risk_off", "confidence": 0.5, "summary": "x"},
                  [{"symbol": "SOXS", "conviction": 1.0, "reasoning": "test"}])
    res = engine.execute(s, review="file", source="synthetic")
    assert [r for r in res if r["signal"] == "HEDGE"][0]["outcome"] == "hedge_rejected"

    # 名单外的标的：拒绝
    _write_review(path, {"regime": "risk_off", "confidence": 0.9, "summary": "x"},
                  [{"symbol": "NVDA", "conviction": 1.0, "reasoning": "test"}])
    res = engine.execute(s, review="file", source="synthetic")
    assert [r for r in res if r["signal"] == "HEDGE"][0]["outcome"] == "hedge_rejected"

    # 偏空 + 高置信度：开仓，金额 ≤ 本金 3%，同时做多买入被拦截
    approve = [{"symbol": b, "action": "APPROVE", "confidence": 0.9, "conviction": 1, "reasoning": "x"} for b in buys]
    _write_review(path, {"regime": "risk_off", "confidence": 0.85, "summary": "AI 融资担忧"},
                  [{"symbol": "SOXS", "conviction": 1.0, "reasoning": "对冲"}], approve)
    res = engine.execute(s, review="file", source="synthetic")
    h = [r for r in res if r["signal"] == "HEDGE"][0]
    assert h["outcome"] == "filled", h.get("reason")
    assert h["order"]["quantity"] * h["order"]["price"] <= 100000 * 0.03
    assert not [r for r in res if r["signal"] == "BUY" and r["symbol"] in buys]
    assert "SOXS" in journal.hedges("paper")


def test_hedge_exits_when_view_changes(tmp_path, monkeypatch):
    import json
    from datetime import date, timedelta
    engine = _isolate(tmp_path, monkeypatch)
    import wquant.journal as journal
    import wquant.broker as broker
    s = _hedge_settings()
    # 模拟已有对冲仓位
    pb = broker.PaperBroker({}, 100000)
    from wquant.risk import OrderIntent
    pb.submit(OrderIntent("SOXS", "BUY", 10, 50.0))
    journal.set_hedge("paper", "SOXS", {"entry_date": str(date.today()), "max_hold_days": 10})
    # 昨天的研判已转为中性
    y = engine.DECISIONS / str(date.today() - timedelta(days=1))
    y.mkdir(parents=True)
    (y / "decisions.json").write_text(json.dumps({"date": y.name, "market_view":
                                                  {"regime": "neutral", "confidence": 0.6, "summary": "企稳"}}))
    path = engine.scan(s, source="synthetic")
    packet = json.loads(path.read_text())
    sells = [c for c in packet["candidates"] if c["symbol"] == "SOXS" and c["action"] == "SELL"]
    assert sells and "研判" in sells[0]["rule_reason"]


def test_buy_and_hold_handles_late_listing():
    from wquant.backtest import buy_and_hold
    a = synthetic_bars("A", 300, seed=1)
    b = synthetic_bars("B", 100, seed=2)        # 晚上市
    bh = buy_and_hold({"A": a, "B": b}, 100_000)
    assert bh.index[0] == a.index[0] and not bh.isna().any()
