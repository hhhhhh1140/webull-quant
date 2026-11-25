import numpy as np
import pandas as pd

from wquant import swing_study as SS


def _fake_data(n=300):
    idx = pd.bdate_range("2020-01-01", periods=n)
    drift = np.where(np.arange(n) < n // 2, 0.003, -0.004)
    c = 100 * np.exp(np.cumsum(drift))
    sox = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c}, index=idx)
    soxl = pd.DataFrame({"open": c, "high": c * 1.03, "low": c * 0.97, "close": c}, index=idx)
    inv = 10000 / c
    soxs = pd.DataFrame({"open": inv, "high": inv * 1.03, "low": inv * 0.97, "close": inv}, index=idx)
    return {
        "sox": sox, "soxl": soxl, "soxs": soxs,
        "vix": pd.Series(np.where(np.arange(n) < n // 2, 15.0, 28.0), index=idx),
        "irx": pd.Series(2.0, index=idx),
    }


def test_long_and_short_never_overlap_and_tp_triggers():
    data = _fake_data()
    idx = data["sox"].index
    ls = pd.Series(0, index=idx)
    ss = pd.Series(0, index=idx)
    ls.iloc[10:40] = 1
    ss.iloc[200:250] = 1
    eq, tr = SS.simulate(data, ls, ss, s_stop=0.12, s_tp=0.10, s_max_days=20)
    assert len(eq) == len(idx)
    assert (tr.side == "SOXL").any() and (tr.side == "SOXS").any()
    # 做空段持续下跌，SOXS 应触发止盈离场而不是拿到信号结束
    s_tr = tr[tr.side == "SOXS"].iloc[0]
    assert s_tr["reason"] == "tp" and s_tr["ret"] >= 0.09


def test_build_strategies_mutually_exclusive():
    data = _fake_data()
    for name, (ls, ss, _) in SS.build_strategies(data).items():
        # 上涨信号和下跌信号不能同时为 1
        assert not ((ls == 1) & (ss == 1)).any(), name
