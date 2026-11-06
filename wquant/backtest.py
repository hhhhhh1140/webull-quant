"""日线组合回测。

- 第 t 天收盘后计算信号，第 t+1 天开盘成交（避免未来函数）
- 仓位计算使用与实盘相同的 RiskManager
- 只回测规则层。AI 层无法做历史回测：模型训练数据里已经包含了这些历史行情和新闻，
  回测结果会严重高估（前视偏差）。AI 层应在沙盒里前向测试来验证。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import add_indicators
from .risk import RiskManager
from .strategy import entry_ok, exit_reason, index_weak, inverse_entry_ok, inverse_exit_reason, market_regime


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    qty: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    reason: str = ""

    @property
    def pnl(self) -> float:
        return 0.0 if self.exit_price is None else (self.exit_price - self.entry_price) * self.qty


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade]
    metrics: dict = field(default_factory=dict)


def run_backtest(bars: dict[str, pd.DataFrame], strat: dict, risk_cfg: dict, bt: dict,
                 universe: list[str] | None = None, groups: dict[str, str] | None = None,
                 instruments: dict[str, dict] | None = None) -> BacktestResult:
    """universe 为 None 时交易 bars 里的全部股票；大盘过滤用的指数可以在 bars 里但不在 universe 里。"""
    all_frames = {s: add_indicators(df, strat) for s, df in bars.items() if len(df) > 2}
    frames = {s: f for s, f in all_frames.items() if universe is None or s in universe}
    dates = sorted(set().union(*[df.index for df in frames.values()]))
    rm = RiskManager(risk_cfg, list(frames))
    groups = groups or {}
    inst = lambda s: (instruments or {}).get(s) or {"leverage": 1, "inverse": False}
    slip = bt.get("slippage_bps", 5) / 1e4
    fee = bt.get("commission_per_trade", 0.0)

    cash = float(bt.get("initial_cash", 100_000))
    pos: dict[str, Trade] = {}
    trail: dict[str, float] = {}
    pending: list[tuple[str, str, str]] = []   # (symbol, side, reason)，次日开盘执行
    closed: list[Trade] = []
    curve = []
    prev_equity = cash

    for d in dates:
        # 1) 执行昨天收盘产生的订单（按今日开盘价）
        for sym, side, reason in pending:
            df = frames[sym]
            if d not in df.index:
                continue
            px = df.at[d, "open"]
            if side == "SELL" and sym in pos:
                t = pos.pop(sym)
                t.exit_date, t.exit_price, t.reason = d, px * (1 - slip), reason
                cash += t.exit_price * t.qty - fee
                closed.append(t)
                trail.pop(sym, None)
            elif side == "BUY" and sym not in pos:
                equity = cash + sum(frames[s].at[d, "open"] * p.qty if d in frames[s].index else p.entry_price * p.qty
                                    for s, p in pos.items())
                g = groups.get(sym)
                gc = sum(1 for s in pos if groups.get(s) == g)
                dec = rm.size_buy(sym, px * (1 + slip), equity, cash, len(pos), start_of_day_equity=prev_equity,
                                  leverage=inst(sym)["leverage"], group=g, group_count=gc, fractional=True)
                if dec.approved:
                    fill = px * (1 + slip)
                    q = dec.intent.quantity
                    cash -= fill * q + fee
                    pos[sym] = Trade(sym, d, fill, q, reason=reason)
                    trail[sym] = fill
        pending = []

        # 2) 收盘：盯市 + 生成明日信号
        risk_off = market_regime(all_frames, strat, at=d)["risk_off"]
        weak_short = index_weak(all_frames, strat, at=d)
        mv = 0.0
        for sym, df in frames.items():
            if d not in df.index:
                if sym in pos:
                    mv += pos[sym].entry_price * pos[sym].qty
                continue
            row = df.loc[d]
            if sym in pos:
                mv += row["close"] * pos[sym].qty
                trail[sym] = max(trail[sym], row["close"])
                if not np.isnan(row["atr"]):
                    if inst(sym)["inverse"]:
                        held_days = int(((df.index > pos[sym].entry_date) & (df.index <= d)).sum())
                        r = inverse_exit_reason(row, strat, trail[sym], held_days)
                    else:
                        r = exit_reason(row, strat, trail[sym])
                    if r:
                        pending.append((sym, "SELL", r))
            elif inst(sym)["inverse"]:
                if inverse_entry_ok(row, strat, weak_short):
                    pending.append((sym, "BUY", "inverse"))
            elif not risk_off and entry_ok(row, strat):
                pending.append((sym, "BUY", "entry"))
        # 卖出先于买入执行，释放资金
        pending.sort(key=lambda x: x[1] != "SELL")
        prev_equity = cash + mv
        curve.append((d, prev_equity))

    # 收尾：按最后收盘价平掉剩余仓位以统计
    last = dates[-1]
    for sym, t in pos.items():
        t.exit_date, t.exit_price, t.reason = last, frames[sym]["close"].iloc[-1], "回测结束"
        closed.append(t)

    equity = pd.Series(dict(curve)).sort_index()
    return BacktestResult(equity, closed, compute_metrics(equity, closed))


def compute_metrics(equity: pd.Series, trades: list[Trade]) -> dict:
    rets = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    dd = equity / equity.cummax() - 1
    wins = [t for t in trades if t.pnl > 0]
    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return": round(float(total), 4),
        "cagr": round(float((1 + total) ** (1 / years) - 1), 4),
        "sharpe": round(float(rets.mean() / rets.std() * np.sqrt(252)), 2) if rets.std() > 0 else 0.0,
        "max_drawdown": round(float(dd.min()), 4),
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3) if trades else 0.0,
        "avg_pnl": round(float(np.mean([t.pnl for t in trades])), 2) if trades else 0.0,
    }


def buy_and_hold(bars: dict[str, pd.DataFrame], initial_cash: float) -> pd.Series:
    """等权买入持有基准。"""
    closes = pd.DataFrame({s: df["close"] for s, df in bars.items()}).sort_index().ffill()
    # 每只按自己的首个有效价格归一化；上市较晚的（如 SNDK）从上市那天起加入平均
    first = closes.apply(lambda c: c.loc[c.first_valid_index()])
    norm = closes / first
    return norm.mean(axis=1, skipna=True).dropna() * initial_cash
