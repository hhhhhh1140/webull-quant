"""2020 起 SOXL / SOXS 波段买卖点回测。

在项目现有 wquant/swing.py（Donchian / EMA / ENSEMBLE）和 wquant/strategy.py（反向 ETF 短线止损与持有上限）
的基础上，比较「对称波段」「快波段」「不对称波段（中波段做多 + 恐慌破位短空止盈）」的买卖点与收益表现。

核心约束：
  - 所有信号基于 1 倍指数 ^SOX 与 ^VIX（第 t 天收盘出信号，第 t+1 天开盘成交）
  - 上涨趋势不做空，下跌趋势不做多
  - SOXL / SOXS 使用复权价，单边成本 0.10%，空仓现金按 13 周美债收益率计息
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import regime_data
from .swing import SwingParams, ensemble_weight, target_series

START = "2020-01-01"
SLIP = 0.0010


def _adj_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    r = df.adjclose / df.close
    out = pd.DataFrame(
        {"open": df.open * r, "high": df.high * r, "low": df.low * r, "close": df.adjclose},
        index=df.index,
    ).dropna()
    return out[out.open > 0]


def load_frames(d: dict | None = None) -> dict[str, pd.DataFrame | pd.Series]:
    d = d or regime_data.load()
    sox = d["SOX"][["open", "high", "low", "close"]].dropna()
    return {
        "sox": sox,
        "soxl": _adj_ohlc(d["SOXL"]),
        "soxs": _adj_ohlc(d["SOXS"]),
        "vix": d["VIX"].close.reindex(sox.index).ffill(),
        "irx": d["IRX"].close.reindex(sox.index).ffill(),
    }


def simulate(
    data: dict,
    long_sig: pd.Series,
    short_sig: pd.Series,
    *,
    start: str = START,
    alloc_l: float = 0.50,
    alloc_s: float = 0.35,
    slip: float = SLIP,
    l_stop: float | None = None,
    s_stop: float | None = None,
    s_tp: float | None = None,
    s_max_days: int | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """逐日波段回测：第 t 天收盘信号，第 t+1 天开盘成交；盘中止损/止盈按保守顺序（先止损后止盈）模拟。"""
    sox, soxl, soxs, irx = data["sox"], data["soxl"], data["soxs"], data["irx"]
    dates = sox.index.intersection(soxl.index).intersection(soxs.index)
    dates = dates[dates >= pd.Timestamp(start)]
    inst = {1: soxl, -1: soxs}

    cash, side, qty, entry_px, entry_day, hold, cooldown = 100_000.0, 0, 0.0, 0.0, None, 0, 0
    eq, trades = [], []

    for k, dt in enumerate(dates):
        if k > 0:
            pdt = dates[k - 1]
            ls, ss = int(long_sig.loc[pdt]), int(short_sig.loc[pdt])
            want = 1 if (ls and not ss) else (-1 if (ss and not ls) else 0)
            if cooldown and want != cooldown:
                cooldown = 0
            if want == cooldown:
                want = 0
            hit_max = side == -1 and s_max_days is not None and hold >= s_max_days
            if hit_max:
                want, cooldown = 0, -1

            if want != side:
                if side != 0:
                    px = float(inst[side].open.loc[dt]) * (1 - slip)
                    cash += qty * px
                    trades.append({
                        "side": "SOXL" if side == 1 else "SOXS",
                        "entry": entry_day.strftime("%Y-%m-%d"),
                        "exit": dt.strftime("%Y-%m-%d"),
                        "entry_px": round(entry_px, 2),
                        "exit_px": round(px, 2),
                        "ret": px / entry_px - 1,
                        "days": hold,
                        "reason": "max_days" if hit_max else "signal",
                    })
                    side, qty, hold = 0, 0.0, 0
                if want != 0:
                    al = alloc_l if want == 1 else alloc_s
                    px = float(inst[want].open.loc[dt]) * (1 + slip)
                    qty = cash * al / px
                    cash -= qty * px
                    side, entry_px, entry_day, hold = want, px, dt, 0

        if side != 0:
            hold += 1
            bar = inst[side].loc[dt]
            st = l_stop if side == 1 else s_stop
            tp = None if side == 1 else s_tp
            if st and bar.low <= entry_px * (1 - st):
                px = min(float(bar.open), entry_px * (1 - st)) * (1 - slip)
                cash += qty * px
                trades.append({
                    "side": "SOXL" if side == 1 else "SOXS",
                    "entry": entry_day.strftime("%Y-%m-%d"),
                    "exit": dt.strftime("%Y-%m-%d"),
                    "entry_px": round(entry_px, 2),
                    "exit_px": round(px, 2),
                    "ret": px / entry_px - 1,
                    "days": hold,
                    "reason": "stop",
                })
                cooldown, side, qty, hold = side, 0, 0.0, 0
            elif tp and bar.high >= entry_px * (1 + tp):
                px = max(float(bar.open), entry_px * (1 + tp)) * (1 - slip)
                cash += qty * px
                trades.append({
                    "side": "SOXL" if side == 1 else "SOXS",
                    "entry": entry_day.strftime("%Y-%m-%d"),
                    "exit": dt.strftime("%Y-%m-%d"),
                    "entry_px": round(entry_px, 2),
                    "exit_px": round(px, 2),
                    "ret": px / entry_px - 1,
                    "days": hold,
                    "reason": "tp",
                })
                cooldown, side, qty, hold = side, 0, 0.0, 0

        r_cash = max(float(irx.loc[dt]), 0.0) / 100 / 252
        cash *= 1 + r_cash
        mv = qty * float(inst[side].close.loc[dt]) if side else 0.0
        eq.append((dt, cash + mv))

    if side != 0:
        dt = dates[-1]
        px = float(inst[side].close.loc[dt])
        trades.append({
            "side": "SOXL" if side == 1 else "SOXS",
            "entry": entry_day.strftime("%Y-%m-%d"),
            "exit": dt.strftime("%Y-%m-%d") + "*",
            "entry_px": round(entry_px, 2),
            "exit_px": round(px, 2),
            "ret": px / entry_px - 1,
            "days": hold,
            "reason": "open",
        })
    return pd.Series(dict(eq)), pd.DataFrame(trades)


def build_strategies(data: dict) -> dict[str, tuple[pd.Series, pd.Series, dict]]:
    sox, vix = data["sox"], data["vix"]
    c = sox.close
    z = pd.Series(0, index=c.index, dtype=int)
    ema5 = c.ewm(span=5, adjust=False).mean()
    ema10 = c.ewm(span=10, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema30 = c.ewm(span=30, adjust=False).mean()
    sma50 = c.rolling(50).mean()

    don_both = target_series(sox, SwingParams("donchian", n_entry=20, n_exit=10))
    don_l = (don_both == 1).astype(int)
    don_s = (don_both == -1).astype(int)
    ens_l = (ensemble_weight(sox) >= 2 / 3).astype(int)
    ema_fast_l = ((c > ema10) & (c > ema20) & (ema5 > ema10)).astype(int)
    ema_fast_s = ((c < ema10) & (c < ema20) & (ema5 < ema10)).astype(int)
    # 恐慌破位短空：跌破 50 日线 + 短期均线空头排列 + VIX 升温
    panic_s20 = ((c < sma50) & (ema10 < ema30) & (vix > 20)).astype(int)
    panic_s25 = ((c < sma50) & (ema10 < ema30) & (vix > 25)).astype(int)

    return {
        "持有SOXL(50%仓)": (pd.Series(1, index=c.index), z, dict(alloc_l=0.50, alloc_s=0.0)),
        "[现有] Donchian(20,10) 多空对称": (don_l, don_s, dict(alloc_l=0.50, alloc_s=0.50)),
        "[现有] Donchian(20,10) 只做多": (don_l, z, dict(alloc_l=0.50, alloc_s=0.0)),
        "[现有] ENSEMBLE 只做多": (ens_l, z, dict(alloc_l=0.50, alloc_s=0.0)),
        "快波段 EMA(5,10,20) 多空(空止盈25%)": (ema_fast_l, ema_fast_s, dict(alloc_l=0.50, alloc_s=0.35, s_stop=0.12, s_tp=0.25, s_max_days=15)),
        "不对称: Donchian做多 + 破50线/VIX>20短空": (don_l, panic_s20, dict(alloc_l=0.50, alloc_s=0.35, s_stop=0.12, s_tp=0.30, s_max_days=20)),
        "不对称: Donchian做多 + 破50线/VIX>25短空": (don_l, panic_s25, dict(alloc_l=0.50, alloc_s=0.35, s_stop=0.10, s_tp=0.30, s_max_days=20)),
        "不对称: ENSEMBLE做多 + 破50线/VIX>20短空": (ens_l, panic_s20, dict(alloc_l=0.50, alloc_s=0.35, s_stop=0.12, s_tp=0.30, s_max_days=20)),
    }


def summary_row(name: str, eq: pd.Series, tr: pd.DataFrame) -> dict:
    def seg(a: str, b: str | None = None):
        s = eq.loc[a:b]
        yrs = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
        tot = s.iloc[-1] / s.iloc[0] - 1
        r = s.pct_change().dropna()
        sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
        return (1 + tot) ** (1 / yrs) - 1, (s / s.cummax() - 1).min(), sh

    ca, dda, sha = seg(START)
    c1, _, _ = seg(START, "2023-12-31")
    c2, _, _ = seg("2024-01-01")
    L = tr[tr.side == "SOXL"] if len(tr) else pd.DataFrame()
    S = tr[tr.side == "SOXS"] if len(tr) else pd.DataFrame()
    return {
        "策略": name, "全期年化": ca, "最大回撤": dda, "夏普": sha,
        "20-23": c1, "24-26": c2,
        "多笔": len(L), "多均盈": L.ret.mean() if len(L) else 0.0,
        "空笔": len(S), "空均盈": S.ret.mean() if len(S) else 0.0,
        "均持天": tr.days.mean() if len(tr) else 0.0,
    }


def main(fetch: bool = False, since: str = "2026-08-01") -> None:
    pd.set_option("display.width", 220, "display.max_columns", 20)
    if fetch:
        regime_data.fetch_all()
    data = load_frames()
    strats = build_strategies(data)
    rows, res = [], {}
    for name, (ls, ss, kw) in strats.items():
        eq, tr = simulate(data, ls, ss, **kw)
        res[name] = (eq, tr)
        rows.append(summary_row(name, eq, tr))

    df = pd.DataFrame(rows).set_index("策略")
    fmt = {"全期年化": "{:+.1%}", "最大回撤": "{:.1%}", "夏普": "{:.2f}",
           "20-23": "{:+.1%}", "24-26": "{:+.1%}", "多均盈": "{:+.1%}", "空均盈": "{:+.1%}", "均持天": "{:.1f}"}
    print("== 2020-2026 SOXL / SOXS 波段策略对比（做多50%仓，做空35%仓，单边0.1%成本）==")
    print(df.apply(lambda col: col.map(lambda v, c=col.name: fmt.get(c, "{}").format(v))).to_string())

    print(f"\n== {since} 以来的实际买卖点 ==")
    for name in (
        "[现有] Donchian(20,10) 只做多",
        "快波段 EMA(5,10,20) 多空(空止盈25%)",
        "不对称: Donchian做多 + 破50线/VIX>20短空",
        "不对称: Donchian做多 + 破50线/VIX>25短空",
    ):
        _, tr = res[name]
        sub = tr[tr.entry >= since].copy() if len(tr) else pd.DataFrame()
        print(f"\n--- {name} ---")
        if len(sub):
            sub["ret"] = sub["ret"].map(lambda v: f"{v:+.1%}")
            print(sub.to_string(index=False))
        else:
            print("  无交易")
