"""三层每日公允价值系统的单元测试（含无未来函数检验、事件降权与回测验证）。"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from wquant.fair_value import (
    backtest_fair_value,
    build_stat_features,
    compute_daily_fair_values,
    compute_group_fundamental_medians,
    estimate_fundamental_layer,
    estimate_peer_layer,
    estimate_stat_layer,
)
from wquant.fair_value_data import parse_yahoo_fundamental
from wquant.fair_value_report import format_terminal_table, save_daily_report


def _make_correlated_bars(n: int = 360, seed: int = 42) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """构造带真实板块联动的合成日线，便于精确验证第一层和第二层逻辑。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-02", periods=n)
    mkt_ret = rng.normal(0.0004, 0.01, n)
    semi_factor = rng.normal(0.0005, 0.015, n)

    def _to_ohlcv(log_p: np.ndarray) -> pd.DataFrame:
        close = 100.0 * np.exp(log_p)
        open_ = close * (1.0 + rng.normal(0.0, 0.004, n))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.008, n)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.008, n)))
        vol = rng.integers(2_000_000, 10_000_000, n).astype(float)
        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)

    spy_lp = np.cumsum(mkt_ret)
    qqq_lp = np.cumsum(1.15 * mkt_ret + rng.normal(0, 0.003, n))
    # 三只高度相关的半导体股票
    a_lp = np.cumsum(0.8 * mkt_ret + 1.0 * semi_factor + rng.normal(0, 0.004, n))
    b_lp = np.cumsum(0.8 * mkt_ret + 1.0 * semi_factor + rng.normal(0, 0.004, n))
    c_lp = np.cumsum(0.8 * mkt_ret + 1.0 * semi_factor + rng.normal(0, 0.004, n))

    bars = {
        "SPY": _to_ohlcv(spy_lp),
        "QQQ": _to_ohlcv(qqq_lp),
        "AAA": _to_ohlcv(a_lp),
        "BBB": _to_ohlcv(b_lp),
        "CCC": _to_ohlcv(c_lp),
    }
    groups = {"SPY": "index", "QQQ": "index", "AAA": "semis", "BBB": "semis", "CCC": "semis"}
    return bars, groups


def test_peer_layer_detects_dislocation_and_premarket_shift():
    bars, groups = _make_correlated_bars(n=200, seed=7)
    close_panel = pd.DataFrame({s: df["close"] for s, df in bars.items()})

    # 正常收盘价应接近公允价，R² 很高
    last_prices = {s: float(close_panel[s].iloc[-1]) for s in close_panel.columns}
    base = estimate_peer_layer("AAA", "semis", groups, close_panel, ref_price=last_prices["AAA"], live_prices=last_prices)
    assert base["peer_r2"] > 0.85

    # 模拟盘前：AAA 没动，但同板块 BBB/CCC 与大盘全涨 4% → AAA 的板块联动公允价应显著抬升
    rally = dict(last_prices)
    for s in ("SPY", "QQQ", "BBB", "CCC"):
        rally[s] = last_prices[s] * 1.04
    shifted = estimate_peer_layer("AAA", "semis", groups, close_panel, ref_price=last_prices["AAA"], live_prices=rally)
    assert shifted["peer_fv"] > base["peer_fv"] * 1.02
    assert shifted["peer_mispricing_pct"] < -2.0
    assert shifted["peer_z"] < -1.0


def test_stat_layer_strictly_causal_no_lookahead():
    bars, _ = _make_correlated_bars(n=340, seed=11)
    df1 = bars["AAA"].copy()
    df2 = bars["AAA"].copy()
    qqq = bars["QQQ"]["close"]

    eval_pos = 300
    # 极端篡改 eval_pos 之后的所有未来价格，验证不会影响 eval_pos 当日的特征与预测
    df2.iloc[eval_pos + 1 :, df2.columns.get_loc("close")] *= 3.5
    df2.iloc[eval_pos + 1 :, df2.columns.get_loc("high")] *= 3.5
    df2.iloc[eval_pos + 1 :, df2.columns.get_loc("low")] *= 3.5

    f1 = build_stat_features(df1, qqq)
    f2 = build_stat_features(df2, qqq)
    assert np.allclose(f1.iloc[eval_pos].to_numpy(), f2.iloc[eval_pos].to_numpy(), equal_nan=True)

    r1 = estimate_stat_layer(df1, qqq, eval_pos=eval_pos)
    r2 = estimate_stat_layer(df2, qqq, eval_pos=eval_pos)
    assert r1["stat_fv"] == r2["stat_fv"]
    assert r1["stat_pred_ret_pct"] == r2["stat_pred_ret_pct"]


def test_fundamental_layer_and_earnings_repricing_downweight():
    bars, groups = _make_correlated_bars(n=320, seed=19)
    p_aaa = float(bars["AAA"]["close"].iloc[-1])
    p_bbb = float(bars["BBB"]["close"].iloc[-1])
    p_ccc = float(bars["CCC"]["close"].iloc[-1])

    fundamentals = {
        "SPY": {"symbol": "SPY", "quote_type": "ETF", "ref_price": float(bars["SPY"]["close"].iloc[-1]), "ref_source": "regular"},
        "QQQ": {"symbol": "QQQ", "quote_type": "ETF", "ref_price": float(bars["QQQ"]["close"].iloc[-1]), "ref_source": "regular"},
        "AAA": {
            "symbol": "AAA", "quote_type": "EQUITY", "ref_price": p_aaa, "ref_source": "pre_market",
            "forward_pe": 15.0, "forward_eps": p_aaa / 15.0, "revenue_growth": 0.30,
            "target_median": p_aaa * 1.25, "analyst_count": 20,
        },
        "BBB": {
            "symbol": "BBB", "quote_type": "EQUITY", "ref_price": p_bbb, "ref_source": "pre_market",
            "forward_pe": 25.0, "forward_eps": p_bbb / 25.0, "revenue_growth": 0.20,
            "target_median": p_bbb * 1.05, "analyst_count": 15,
        },
        "CCC": {
            "symbol": "CCC", "quote_type": "EQUITY", "ref_price": p_ccc, "ref_source": "pre_market",
            "forward_pe": 26.0, "forward_eps": p_ccc / 26.0, "revenue_growth": 0.18,
            "target_median": p_ccc * 1.02, "analyst_count": 12,
        },
    }

    gm, glob = compute_group_fundamental_medians(groups, fundamentals)
    # ETF 应跳过第 3 层
    assert estimate_fundamental_layer("SPY", "index", fundamentals["SPY"]["ref_price"], fundamentals["SPY"], gm, glob) == {}

    # AAA 估值明显低于同行且增速更高，基本面锚应显著高于现价，但战术引力价有界（不会一步拉偏 25%）
    f_aaa = estimate_fundamental_layer("AAA", "semis", p_aaa, fundamentals["AAA"], gm, glob)
    assert f_aaa["fund_fv"] > p_aaa * 1.15
    assert p_aaa < f_aaa["fund_tactical_fv"] < p_aaa * 1.06

    # 验证财报刚落地（POST_EARNINGS_REPRICING）会自动降低第 1 层权重、提升第 3 层权重
    normal_res = {
        r.symbol: r for r in compute_daily_fair_values(bars, groups, fundamentals=fundamentals)
    }
    ev_ctx = {
        "macro_alerts": ["CPI (2026-11-26)"],
        "earnings_flags": {"AAA": {"status": "POST_EARNINGS_REPRICING", "date": str(date.today()), "bdays": -1}},
    }
    post_res = {
        r.symbol: r for r in compute_daily_fair_values(bars, groups, fundamentals=fundamentals, event_context=ev_ctx)
    }
    assert post_res["AAA"].weights["peer"] < normal_res["AAA"].weights["peer"]
    assert post_res["AAA"].weights["fund"] > normal_res["AAA"].weights["fund"]
    assert "POST_EARNINGS_REPRICING" in post_res["AAA"].flags
    assert "MACRO_EVENT" in post_res["AAA"].flags


def test_parse_yahoo_fundamental_premarket_priority():
    sample = {
        "price": {
            "quoteType": "EQUITY",
            "marketState": "PRE",
            "regularMarketPrice": {"raw": 100.0},
            "regularMarketPreviousClose": {"raw": 99.0},
            "preMarketPrice": {"raw": 103.5},
            "marketCap": {"raw": 5e11},
        },
        "summaryDetail": {"forwardPE": {"raw": 22.5}, "priceToSalesTrailing12Months": {"raw": 8.2}},
        "defaultKeyStatistics": {"forwardEps": {"raw": 4.6}},
        "financialData": {"targetMedianPrice": {"raw": 125.0}, "numberOfAnalystOpinions": {"raw": 32}, "revenueGrowth": {"raw": 0.28}},
        "calendarEvents": {"earnings": {"earningsDate": [{"fmt": "2026-12-01"}]}},
    }
    parsed = parse_yahoo_fundamental("NVDA", sample)
    assert parsed["ref_price"] == 103.5
    assert parsed["ref_source"] == "pre_market"
    assert parsed["forward_pe"] == 22.5
    assert parsed["target_median"] == 125.0
    assert parsed["earnings_dates"] == ["2026-12-01"]


def test_walk_forward_backtest_and_report_save(tmp_path):
    bars, groups = _make_correlated_bars(n=360, seed=29)
    bt = backtest_fair_value(
        bars,
        groups,
        cfg={"backtest_eval_days": 80, "backtest_retrain_every": 10},
    )
    assert "error" not in bt
    assert bt["total_samples"] > 100
    assert 30.0 <= bt["range_calibration"]["in_1s_pct"] <= 95.0
    assert set(bt["layers"].keys()) == {"peer", "stat", "composite"}

    results = compute_daily_fair_values(bars, groups)
    assert len(results) == 5
    for r in results:
        assert r.low_90 < r.low_1s < r.fair_value < r.high_1s < r.high_90

    tbl = format_terminal_table(results)
    assert "AAA" in tbl
    json_p, md_p = save_daily_report(results, backtest_summary=bt, out_dir=tmp_path, report_date="2026-11-25")
    assert json_p.exists() and md_p.exists()
    data = json.loads(json_p.read_text(encoding="utf-8"))
    assert data["count"] == 5
    assert "# 每日美股公允价值报告" in md_p.read_text(encoding="utf-8")


def test_overnight_quotes_and_bear_kospi_regime_protection():
    bars, groups = _make_correlated_bars(n=320, seed=53)
    last_close = {s: float(df["close"].iloc[-1]) for s, df in bars.items()}

    # 模拟滞后卖方目标价大幅看多 AAA（+35%）
    fundamentals = {
        s: {
            "symbol": s,
            "quote_type": "ETF" if s in ("SPY", "QQQ") else "EQUITY",
            "ref_price": last_close[s],
            "ref_source": "post_market",
            "forward_pe": 15.0 if s == "AAA" else 22.0,
            "forward_eps": last_close[s] / (15.0 if s == "AAA" else 22.0),
            "target_median": last_close[s] * 1.35,
            "analyst_count": 25,
        }
        for s in bars
    }

    # 模拟 Webull 夜盘全线下跌 3%
    live_quotes = {
        s: {"price": last_close[s] * 0.97, "session": "OVN", "source": "overnight", "timestamp_et": "2026-11-24T22:30:00-05:00"}
        for s in bars
    }
    base_res = {r.symbol: r for r in compute_daily_fair_values(bars, groups, fundamentals=fundamentals)}
    ovn_res = {
        r.symbol: r
        for r in compute_daily_fair_values(
            bars,
            groups,
            fundamentals=fundamentals,
            live_quotes=live_quotes,
            regime_context={"soxx_votes": -3, "soxx_regime": -1, "qqq_votes": -2, "qqq_regime": -1, "kospi_move_pct": -5.0},
        )
    }

    # 1) 参考价优先采用 Webull 夜盘实价，较昨收约 -3%
    assert ovn_res["AAA"].price_source == "overnight"
    assert abs(ovn_res["AAA"].overnight_chg_pct - (-3.0)) < 0.05
    # 2) 夜盘大盘下跌时，L2 统计公允价同步下移，不再死锚在昨收导致虚假低估
    assert ovn_res["AAA"].stat_fv < base_res["AAA"].stat_fv
    # 3) 熊市 + KOSPI 冲击触发时，滞后基本面看多权重被清零，且打上双风控标签
    assert "BEAR_REGIME" in ovn_res["AAA"].flags
    assert "KOSPI_SHOCK" in ovn_res["AAA"].flags
    assert "fund" not in ovn_res["AAA"].weights
