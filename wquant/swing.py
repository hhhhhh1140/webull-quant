"""波段趋势系统：用 1 倍指数（SOXX）判断方向，持有 3 倍 ETF（SOXL 做多 / SOXS 做空）。

为什么用 SOXX 判断：3 倍 ETF 的日内波动是指数的 3 倍，直接在它上面设均线和止损很容易被噪音洗出去。
在 1 倍指数上判断趋势、设止损，持有 3 倍 ETF 吃趋势。

信号在收盘后算出（第 t 天），第 t+1 天开盘执行（回测）；实盘可以提前到夜盘执行（见 overnight 择时）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd


@dataclass
class SwingParams:
    rule: str = "donchian"        # donchian | ema | sma_mom
    n_entry: int = 20             # donchian：突破过去 N 天最高 / 最低入场
    n_exit: int = 10              # donchian：跌破过去 M 天最低 / 突破最高离场
    fast: int = 10                # ema：快线
    slow: int = 30                # ema：慢线
    sma: int = 50                 # sma_mom：均线
    mom: int = 20                 # sma_mom：动量回看
    atr_stop: float | None = None # 在 SOXX 上的 ATR 跟踪止损倍数；None = 不用
    long_only: bool = False
    allocation: float = 0.30      # 持仓占本金比例（3 倍 ETF，30% ≈ 90% 指数敞口）
    slippage: float = 0.0005


# ---------------------------------------------------------------- 信号（只用到当天及之前的数据）
def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(), (df.low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


ENSEMBLE = (
    SwingParams(rule="donchian", n_entry=20, n_exit=10, long_only=True),
    SwingParams(rule="sma_mom", sma=50, mom=20, long_only=True),
    SwingParams(rule="ema", fast=10, slow=30, long_only=True),
)


def ensemble_weight(idx: pd.DataFrame, members=ENSEMBLE) -> pd.Series:
    """组合：每条规则管 1/N 仓位，返回 0~1 的多头权重（几条规则看多）。"""
    return sum(target_series(idx, m).clip(lower=0) for m in members) / len(members)


def target_series(idx: pd.DataFrame, p: SwingParams) -> pd.Series:
    """每天收盘后的目标方向：+1 持 SOXL，-1 持 SOXS，0 空仓。"""
    c = idx.close
    tgt = pd.Series(0, index=idx.index, dtype=int)
    if p.rule == "ema":
        f, s = c.ewm(span=p.fast, adjust=False).mean(), c.ewm(span=p.slow, adjust=False).mean()
        tgt[:] = np.where(f > s, 1, -1)
        tgt.iloc[: p.slow] = 0
    elif p.rule == "sma_mom":
        sma = c.rolling(p.sma).mean()
        mom = c / c.shift(p.mom) - 1
        tgt[:] = np.where((c > sma) & (mom > 0), 1, np.where((c < sma) & (mom < 0), -1, 0))
    elif p.rule == "donchian":
        hi_in, lo_in = idx.high.rolling(p.n_entry).max().shift(), idx.low.rolling(p.n_entry).min().shift()
        hi_out, lo_out = idx.high.rolling(p.n_exit).max().shift(), idx.low.rolling(p.n_exit).min().shift()
        state = 0
        vals = []
        for i in range(len(c)):
            x = c.iloc[i]
            if np.isnan(hi_in.iloc[i]):
                vals.append(0)
                continue
            if state == 1 and x < lo_out.iloc[i]:
                state = 0
            elif state == -1 and x > hi_out.iloc[i]:
                state = 0
            if state == 0:
                if x > hi_in.iloc[i]:
                    state = 1
                elif x < lo_in.iloc[i]:
                    state = -1
            vals.append(state)
        tgt[:] = vals
    else:
        raise ValueError(p.rule)
    if p.atr_stop:
        tgt = _apply_atr_stop(idx, tgt, p.atr_stop)
    if p.long_only:
        tgt = tgt.clip(lower=0)
    return tgt


def _apply_atr_stop(idx: pd.DataFrame, tgt: pd.Series, k: float) -> pd.Series:
    """在指数上加 ATR 跟踪止损：触发后空仓，直到原始信号换方向或重新确认后再入场。"""
    atr = _atr(idx)
    out, side, ext, stopped = [], 0, None, 0
    for i, (t, raw) in enumerate(tgt.items()):
        c = idx.close.iloc[i]
        if raw != side and raw != stopped:
            side, ext, stopped = raw, c, 0
        elif raw == 0:
            side, stopped = 0, 0
        if side == 1:
            ext = max(ext, c)
            if c < ext - k * atr.iloc[i]:
                stopped, side = 1, 0
        elif side == -1:
            ext = min(ext, c)
            if c > ext + k * atr.iloc[i]:
                stopped, side = -1, 0
        if stopped and raw != stopped:
            stopped = 0
        out.append(side)
    return pd.Series(out, index=tgt.index)


# ---------------------------------------------------------------- 回测
@dataclass
class SwingResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def backtest(idx: pd.DataFrame, long_df: pd.DataFrame, short_df: pd.DataFrame, p: SwingParams,
             capital: float = 100_000, start: str | None = None) -> SwingResult:
    tgt = target_series(idx, p)
    dates = idx.index.intersection(long_df.index).intersection(short_df.index)
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    cash, side, qty, entry_px, entry_day, worst = capital, 0, 0.0, 0.0, None, 0.0
    eq, trades = [], []
    inst = {1: long_df, -1: short_df}
    for k, d in enumerate(dates):
        # 1) 开盘执行昨天收盘的信号
        if k > 0:
            want = int(tgt.loc[dates[k - 1]])
            if want != side:
                if side != 0:
                    px = inst[side].open.loc[d] * (1 - p.slippage)
                    cash += qty * px
                    trades.append({"side": side, "entry_day": entry_day, "exit_day": d, "entry": entry_px,
                                   "exit": px, "ret": px / entry_px - 1, "worst": worst,
                                   "days": int(np.busday_count(entry_day.date(), d.date()))})
                    side, qty = 0, 0.0
                if want != 0:
                    equity_now = cash
                    px = inst[want].open.loc[d] * (1 + p.slippage)
                    qty = equity_now * p.allocation / px
                    cash -= qty * px
                    side, entry_px, entry_day, worst = want, px, d, 0.0
        # 2) 收盘盯市
        mv = qty * inst[side].close.loc[d] if side else 0.0
        if side:
            worst = min(worst, inst[side].low.loc[d] / entry_px - 1)
        eq.append((d, cash + mv))
    if side:
        d = dates[-1]
        px = inst[side].close.loc[d]
        trades.append({"side": side, "entry_day": entry_day, "exit_day": d, "entry": entry_px, "exit": px,
                       "ret": px / entry_px - 1, "worst": worst, "days": int(np.busday_count(entry_day.date(), d.date())),
                       "open": True})
    equity = pd.Series(dict(eq))
    tr = pd.DataFrame(trades)
    return SwingResult(equity, tr, metrics(equity, tr))


def backtest_weight(idx: pd.DataFrame, long_df: pd.DataFrame, weight: pd.Series, allocation: float = 0.30,
                    slippage: float = 0.0005, capital: float = 100_000, start: str | None = None,
                    rebalance_band: float = 0.05) -> SwingResult:
    """按权重持有 SOXL：目标市值 = 权益 × allocation × weight。权重变化超过 band 才调仓，减少来回交易。"""
    dates = idx.index.intersection(long_df.index)
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    cash, qty, cur_w = capital, 0.0, 0.0
    eq, trades = [], []
    for k, d in enumerate(dates):
        if k > 0:
            w = float(weight.loc[dates[k - 1]])
            if abs(w - cur_w) >= rebalance_band or (w == 0 and cur_w > 0):
                px_o = long_df.open.loc[d]
                equity_now = cash + qty * px_o
                target_qty = equity_now * allocation * w / px_o
                dq = target_qty - qty
                fill = px_o * (1 + slippage if dq > 0 else 1 - slippage)
                cash -= dq * fill
                trades.append({"day": d, "from": cur_w, "to": w, "px": fill})
                qty, cur_w = target_qty, w
        eq.append((d, cash + qty * long_df.close.loc[d]))
    equity = pd.Series(dict(eq))
    tr = pd.DataFrame(trades)
    m = metrics(equity, pd.DataFrame())
    m["调仓次数"] = len(tr)
    return SwingResult(equity, tr, m)


def metrics(equity: pd.Series, tr: pd.DataFrame) -> dict:
    r = equity.pct_change().dropna()
    yrs = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    tot = equity.iloc[-1] / equity.iloc[0] - 1
    out = {
        "总收益": round(float(tot), 3),
        "年化": round(float((1 + tot) ** (1 / yrs) - 1), 3),
        "最大回撤": round(float((equity / equity.cummax() - 1).min()), 3),
        "夏普": round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() > 0 else 0,
        "笔数": int(len(tr)),
    }
    if len(tr):
        out.update({
            "胜率": round(float((tr.ret > 0).mean()), 2),
            "平均持有天数": round(float(tr.days.mean()), 1),
            "单笔平均": round(float(tr.ret.mean()), 3),
            "最好一笔": round(float(tr.ret.max()), 3),
            "最差一笔": round(float(tr.ret.min()), 3),
            "持仓中最大浮亏(中位数)": round(float(tr.worst.median()), 3),
        })
    return out


def yearly(equity: pd.Series) -> dict:
    y = equity.groupby(equity.index.year).agg(["first", "last"])
    prev = None
    out = {}
    for yr, row in y.iterrows():
        base = prev if prev is not None else row["first"]
        out[int(yr)] = round(float(row["last"] / base - 1), 3)
        prev = row["last"]
    return out
