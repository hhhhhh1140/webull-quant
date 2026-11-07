import numpy as np
import pandas as pd
import pytest

from wquant.intraday_bt import Params, load_days, run


def _day(prices_l, prices_s=None, day="2026-11-04", gap_at=None):
    n = len(prices_l)
    idx = pd.date_range(f"{day} 09:30", periods=n, freq="5min", tz="America/New_York")
    def frame(px):
        px = np.asarray(px, float)
        o = np.r_[px[0], px[:-1]]
        if gap_at is not None:
            o = o.copy()
        f = pd.DataFrame({"open": o, "high": np.maximum(o, px) * 1.001, "low": np.minimum(o, px) * 0.999,
                          "close": px, "volume": 1000.0, "session": "RTH"}, index=idx)
        return f
    L = frame(prices_l)
    S = frame(prices_s if prices_s is not None else 100 * 100 / np.asarray(prices_l))
    return L, S


def always_long(day, i, side, st):
    return 1


def test_entry_next_open_and_eod_exit():
    L, S = _day(np.linspace(100, 110, 78))
    days = load_days(L, S)
    p = Params(slippage=0, first_entry="09:30", stop_pct=0.5)
    trades, daily = run(days, always_long, p)
    t = trades[0]
    assert t.entry_t == L.index[1] and t.entry == pytest.approx(L.open.iloc[1])
    assert t.reason == "收盘平仓" and t.exit == pytest.approx(L.close.iloc[-1])
    assert daily.iloc[0] == pytest.approx(t.pnl)


def test_stop_and_gap_through_stop():
    px = [100.0] * 10 + [99.0, 90.0] + [90.0] * 66      # 第 11 根跳空到 90，越过 3% 止损
    L, S = _day(px)
    L.loc[L.index[11], "open"] = 90.0
    days = load_days(L, S)
    p = Params(slippage=0, first_entry="09:30", max_trades=1)
    trades, _ = run(days, always_long, p)
    t = trades[0]
    assert t.reason == "止损"
    assert t.exit == pytest.approx(min(90.0, t.entry * 0.97))   # 跳空按开盘价成交，不是止损价


def test_daily_loss_limit_halts():
    px = np.r_[[100.0] * 5, np.linspace(100, 60, 73)]
    L, S = _day(px)
    days = load_days(L, S)
    p = Params(slippage=0, first_entry="09:30", notional=100_000, daily_loss_limit=0.015, max_trades=10)
    trades, daily = run(days, always_long, p)
    assert len(trades) == 1       # 第一笔止损亏 3% 本金 > 1.5%，当天停止


def test_split_adjust_snaps_to_common_ratio():
    from wquant.intraday_data import adjust_splits
    idx = pd.date_range("2026-07-14 15:00", periods=6, freq="5min", tz="America/New_York")
    px = [8.0, 8.1, 8.2, 77.8, 78.0, 78.5]        # 1:10 合股，同时当天涨了一点
    df = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1000.0, "session": "RTH"}, index=idx)
    df.loc[idx[3], "open"] = 77.8
    out = adjust_splits(df)
    assert out.close.iloc[2] == pytest.approx(82.0)          # ×10，不是 ×9
    assert out.volume.iloc[0] == pytest.approx(100.0)


def _window(prices_l, prices_s, start="2026-11-03 16:00", sessions=None):
    """从前一天 16:00 开始的连续 5 分钟线（简化：都标成 RTH 以外再按时间标注）。"""
    idx = pd.date_range(start, periods=len(prices_l), freq="5min", tz="America/New_York")
    def frame(px):
        px = np.asarray(px, float)
        o = np.r_[px[0], px[:-1]]
        f = pd.DataFrame({"open": o, "high": np.maximum(o, px), "low": np.minimum(o, px), "close": px, "volume": 1.0}, index=idx)
        h = f.index.hour + f.index.minute / 60
        f["session"] = np.where((h >= 9.5) & (h < 16), "RTH", np.where((h >= 4) & (h < 9.5), "PRE", np.where(h >= 20, "OVN", np.where(h < 4, "OVN", "ATH"))))
        return f
    return frame(prices_l), frame(prices_s)


def test_run24_overnight_long_and_reentry():
    from wquant.intraday24 import P24, build_days, run24
    # 两个交易日：11/3 常规时段（为了有昨收），然后 11/3 16:00 → 11/4 16:00
    n_prev = 78
    idx_prev = pd.date_range("2026-11-03 09:30", periods=n_prev, freq="5min", tz="America/New_York")
    n = 24 * 12   # 16:00 → 次日 16:00
    # 16:00-20:00 持平；夜盘 20:00-04:00 涨到 106；之后继续涨到 110
    lp = np.r_[np.full(48, 100.0), np.linspace(100, 106, 96), np.linspace(106, 110, n - 144)]
    l_window, s_window = _window(lp, 10000 / lp)
    prev = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0, "session": "RTH"}, index=idx_prev)
    L = pd.concat([prev, l_window[l_window.index > idx_prev[-1]]])
    S = pd.concat([prev.assign(open=100.0, high=100.0, low=100.0, close=100.0), s_window[s_window.index > idx_prev[-1]]])
    days = build_days(L, S)
    assert len(days) == 1
    p = P24(entry="04:00", exit="16:00", threshold=0.03, stop_pct=0.06, slip_ext=0, slip_rth=0)
    trades, daily = run24(days, p)
    assert trades and trades[0]["side"] == 1               # 夜盘涨超 3% → 做多 SOXL
    assert trades[0]["entry_t"].hour == 4                  # 04:00 判断，下一根开盘成交


def test_multi_session_cap_and_pyramid():
    from wquant.intraday_multi import PM, run_multi
    # 前一天常规时段 100；夜盘涨到 104（21:00 触发，OVN 仓位上限 100%）；盘前 / 常规继续涨到 120
    prev_idx = pd.date_range("2026-11-03 09:30", periods=78, freq="5min", tz="America/New_York")
    win_idx = pd.date_range("2026-11-03 16:05", periods=287, freq="5min", tz="America/New_York")
    def sess(ix):
        h = ix.hour + ix.minute / 60
        return np.where((h >= 9.5) & (h < 16), "RTH", np.where((h >= 4) & (h < 9.5), "PRE", np.where((h >= 16) & (h < 20), "ATH", "OVN")))
    n = len(win_idx)
    px = np.r_[np.full(48, 100.0), np.linspace(100, 104, 12), np.full(84, 104.0), np.linspace(104, 120, n - 144)]
    def frame(prices, idx):
        o = np.r_[prices[0], prices[:-1]]
        return pd.DataFrame({"open": o, "high": np.maximum(o, prices), "low": np.minimum(o, prices), "close": prices,
                             "volume": 1.0, "session": sess(idx)}, index=idx)
    L = pd.concat([frame(np.full(78, 100.0), prev_idx), frame(px, win_idx)])
    S = L.copy(); S[["open", "high", "low", "close"]] = 10000 / L[["open", "high", "low", "close"]]
    base = PM(checkpoints=("21:00",), korea_confirm=None, capital=1000, frac=1.3, slip_ext=0, slip_rth=0, max_frac_ovn=1.0, max_frac=1.3)
    r = run_multi(L, S, None, base)
    t0 = r["trades"].iloc[0]
    assert t0["frac"] == pytest.approx(1.0)              # 夜盘进场被限制在 100%
    r2 = run_multi(L, S, None, base.__class__(**{**base.__dict__, "frac": 0.33, "add_step": 0.03, "add_frac": 0.33}))
    t1 = r2["trades"].iloc[0]
    assert t1["adds"] >= 2 and t1["frac"] <= 1.3 + 1e-9  # 盘前 / 常规时段盈利加仓，总仓位不超过 130%
    assert r2["equity"].iloc[-1] > 1000 * (1 + 0.33 * (120 / 104 - 1))   # 加仓后比只拿初始仓位赚得多
