import numpy as np
import pandas as pd

from wquant import regime_bt as R


def make_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-01", periods=n)
    # 先涨后跌，保证两种趋势都会出现
    drift = np.where(np.arange(n) < n // 2, 0.002, -0.002)
    c = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, n)))
    o = c * (1 + rng.normal(0, 0.002, n))
    px = lambda a: pd.DataFrame({"open": a, "high": a * 1.01, "low": a * .99, "close": a, "adjclose": a}, index=idx)
    empty = px(np.full(n, np.nan)).iloc[:0]
    sox = px(c); sox["open"] = o
    return {"SOX": sox, "SOXL": empty, "SOXS": empty, "VIX": px(np.full(n, 18.0)),
            "IRX": px(np.full(n, 2.0)), "FVX": px(np.full(n, 3.0)), "TNX": px(np.full(n, 3.5))}


def frame():
    f = R.build_frame(make_data())
    return f, R.features(f)


def test_regime_gate_never_longs_in_downtrend_or_shorts_in_uptrend():
    f, x = frame()
    S = R.strategies(f, x)
    for k in ("趋势闸门 多空", "趋势闸门 多空+控波动", "综合打分 多空", "只做多+控波动"):
        t = S[k]
        assert (t[x.regime == -1] <= 0).all()
        assert (t[x.regime == 1] >= 0).all()
        assert (t[x.regime == 0] == 0).all()
    assert x.regime.eq(1).any() and x.regime.eq(-1).any()


def test_signal_is_traded_next_day_only():
    f, _ = frame()
    t = pd.Series(0.0, index=f.index)
    t.iloc[300] = 1.0                       # 第 300 天收盘出信号
    bt = R.run(f, t, cost=0)
    assert bt.pos.iloc[300] == 0 and bt.pos.iloc[301] == 1.0 and bt.pos.iloc[302] == 0
    # 第 301 天：开盘买入，只吃到日内段的 3 倍收益（加现金不计）
    i = 301
    assert np.isclose(bt.ret.iloc[i], f.SOXL_id.iloc[i])
    # 第 302 天：隔夜段仍持有，开盘卖出后只剩现金
    assert np.isclose(bt.ret.iloc[i + 1], (1 + f.SOXL_on.iloc[i + 1]) * (1 + f.cash.iloc[i + 1]) - 1)


def test_costs_charged_on_side_switch():
    f, _ = frame()
    t = pd.Series(1.0, index=f.index)
    t.iloc[400:] = -1.0
    a, b = R.run(f, t, cost=0), R.run(f, t, cost=0.001)
    diff = (a.ret - b.ret).round(10)
    assert np.isclose(diff.iloc[401], 0.002)   # 换边 = 卖 SOXL + 买 SOXS
    assert np.isclose(diff.iloc[1], 0.001)     # 首次建仓


def test_synthetic_leverage_is_three_times_index():
    f, _ = frame()
    assert np.allclose(f.SOXL_on, 3 * f.r_on) and np.allclose(f.SOXS_on, -3 * f.r_on)
