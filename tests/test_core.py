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
