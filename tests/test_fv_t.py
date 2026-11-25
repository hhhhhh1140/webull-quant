"""动态公允价做 T（fv-t）的单元测试：含回测、独立账本隔离、熊市熔断与实盘锁验证。"""
from __future__ import annotations

from datetime import date

import pytest

from wquant.config import Settings
from wquant.fair_value import StockFairValue
from wquant.fv_t import FvTParams, SwingTParams, run_fv_t_backtest, run_swing_t_backtest
from wquant.fv_t_data import synthetic_m5_universe
from wquant.fv_t_live import FvTBook, make_fv_t_broker, plan_fv_t_actions


def _sfv(sym: str, group: str, price: float, fv: float, z: float) -> StockFairValue:
    return StockFairValue(
        symbol=sym,
        group=group,
        as_of_bar_date="2026-11-24",
        ref_price=price,
        last_close=price,
        price_source="overnight",
        overnight_chg_pct=0.0,
        fair_value=fv,
        mispricing_pct=round((price / fv - 1.0) * 100.0, 2),
        z_score=z,
        sigma_1d_pct=3.0,
        low_1s=round(fv * 0.97, 2),
        high_1s=round(fv * 1.03, 2),
        low_90=round(fv * 0.95, 2),
        high_90=round(fv * 1.05, 2),
        signal="小幅低估" if z <= -0.35 else ("小幅高估" if z >= 0.35 else "合理区间"),
    )


def _base_cfg() -> dict:
    return {
        "capital": 10_000.0,
        "notional_per_trade": 2_500.0,
        "max_positions": 4,
        "max_per_group": 2,
        "entry_z": -0.60,
        "exit_z": 0.10,
        "stop_pct": 0.06,
        "max_hold_days": 5,
        "require_soxx_above": -2,
        "kospi_crash_pct": -2.0,
        "daily_loss_pct": 0.03,
    }


def test_plan_buy_low_sell_high_and_group_cap(tmp_path):
    book = FvTBook("uat", capital=10_000.0, path=tmp_path / "book.json")
    # 预先持有 1 只已经回归到公允价上方的标的（应高卖止盈）
    book.record_buy("MU", 4, 600.0, "memory", 620.0, -0.80, "2026-11-20")

    items = [
        _sfv("MU", "memory", 625.0, 620.0, +0.25),      # 回归公允价上方 → SELL
        _sfv("WDC", "memory", 250.0, 265.0, -1.10),     # 明显低估 → BUY #1
        _sfv("STX", "memory", 450.0, 470.0, -0.85),     # 低估 → BUY #2 (memory 达到 max_per_group=2)
        _sfv("SNDK", "memory", 70.0, 75.0, -0.75),      # 同样 memory，但已满 2 只 → 跳过
        _sfv("AMD", "semis", 400.0, 415.0, -0.70),      # 另一板块低估 → BUY #3
        _sfv("AVGO", "semis", 300.0, 315.0, -0.90),     # 财报前禁止买入 → 跳过
    ]
    actions = plan_fv_t_actions(
        fv_items=items,
        book=book,
        regime_ctx={"soxx_votes": 1, "kospi_move_pct": 0.5},
        event_ctx={"earnings_flags": {"AVGO": {"status": "EARNINGS_SOON", "date": "2026-11-26", "bdays": 1}}},
        cfg=_base_cfg(),
        today=date(2026, 11, 25),
    )

    sells = [a for a in actions if a["side"] == "SELL"]
    buys = [a for a in actions if a["side"] == "BUY"]
    assert len(sells) == 1 and sells[0]["symbol"] == "MU" and sells[0]["reason_code"] == "revert_fv"
    buy_syms = [b["symbol"] for b in buys]
    assert buy_syms == ["WDC", "STX", "AMD"]
    assert "SNDK" not in buy_syms
    assert "AVGO" not in buy_syms


def test_bear_regime_liquidates_and_blocks_new_buys(tmp_path):
    book = FvTBook("uat", capital=10_000.0, path=tmp_path / "book.json")
    book.record_buy("MU", 4, 610.0, "memory", 625.0, -0.80, "2026-11-24")

    items = [
        _sfv("MU", "memory", 605.0, 615.0, -0.40),
        _sfv("AMD", "semis", 390.0, 415.0, -1.20),
    ]
    # SOXX = -3 票熊市：已有仓位应触发 bear_regime 卖出，且禁止任何低吸新开仓
    actions = plan_fv_t_actions(
        fv_items=items,
        book=book,
        regime_ctx={"soxx_votes": -3, "kospi_move_pct": -5.5},
        event_ctx={},
        cfg=_base_cfg(),
        today=date(2026, 11, 25),
    )
    assert len(actions) == 1
    assert actions[0]["side"] == "SELL"
    assert actions[0]["symbol"] == "MU"
    assert actions[0]["reason_code"] == "bear_regime"


def test_isolated_book_pnl_and_prod_safety_lock(tmp_path):
    book = FvTBook("uat", capital=10_000.0, path=tmp_path / "book.json")
    book.record_buy("AMD", 5, 400.0, "semis", 415.0, -0.85, "2026-11-20")
    assert book.cash == 8_000.0
    rec = book.record_sell("AMD", 420.0, "revert_fv", "2026-11-24")
    assert rec["pnl"] == 100.0
    assert book.cash == 10_100.0
    assert book.data["realized_pnl"] == 100.0
    assert not book.positions

    # 验证 prod 环境下未开 live_trading_enabled 时拒绝实例化实盘 broker
    prod_settings = Settings(raw={"live_trading_enabled": False}, app_key="k", app_secret="s", environment="prod")
    with pytest.raises(RuntimeError):
        make_fv_t_broker(prod_settings, mode="webull", confirm_live=False, order_cfg={"type": "LIMIT"})


def test_m5_and_swing_t_backtests_run_cleanly():
    syms = ["AAA", "BBB", "CCC", "DDD"]
    groups = {"SPY": "index", "QQQ": "index", "AAA": "semis", "BBB": "semis", "CCC": "memory", "DDD": "memory"}
    m5 = synthetic_m5_universe(syms, groups, days=25, seed=13)
    res_m5 = run_fv_t_backtest(m5, syms, groups, FvTParams(beta_days=5, entry_pct=0.005, entry_z=1.0))
    assert "error" not in res_m5
    assert res_m5["days"] >= 15

    # 合成日线回测（用 test_fair_value 的相关日线构造器）
    from tests.test_fair_value import _make_correlated_bars
    dbars, dgroups = _make_correlated_bars(n=340, seed=21)
    dbars["SOXX"] = dbars["QQQ"].copy()
    res_sw = run_swing_t_backtest(
        dbars, ["AAA", "BBB", "CCC"], dgroups,
        SwingTParams(eval_days=100, entry_z=-0.35, exit_z=0.05),
    )
    assert "error" not in res_sw
    assert res_sw["days"] >= 50
