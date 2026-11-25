"""SOXL / SOXS 常规时段模拟撮合。输入为同一交易日已收盘、按开盘时刻标记的 M5 线。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd


M5 = pd.Timedelta(minutes=5)


@dataclass(frozen=True)
class ScalpParams:
    capital: float = 3000.0
    notional: float = 1000.0
    slippage_bps: float = 3.0
    spread_bps: float = 4.0
    fee_per_order: float = 0.10
    max_trades: int = 12
    lookback: int = 6
    band_std: float = 1.25
    min_band_pct: float = 0.004
    stop_pct: float = 0.008
    target_pct: float = 0.01
    max_hold_bars: int = 6
    daily_loss_pct: float = 0.015

    def __post_init__(self):
        for name in ("capital", "notional", "band_std", "min_band_pct", "stop_pct",
                     "target_pct", "daily_loss_pct"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为正数")
        for name in ("max_trades", "lookback", "max_hold_bars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} 必须为正整数")
        for name in ("slippage_bps", "spread_bps", "fee_per_order"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须为非负数")
        if self.stop_pct >= 1 or self.daily_loss_pct >= 1 or self.fill_drag >= 1:
            raise ValueError("止损、日亏损与单边成交成本必须小于 100%")

    @property
    def fill_drag(self):
        return (self.slippage_bps + self.spread_bps / 2) / 10_000


def run_session(long_df: pd.DataFrame, short_df: pd.DataFrame, params: ScalpParams,
                close_time: pd.Timestamp, trade_from: pd.Timestamp | None = None) -> tuple[list, dict]:
    """重放一日或当日已收盘的前缀。trade_from 指首根允许产生买入信号的 K 线开盘时刻。

    成交时间记录所在 K 线的开盘时刻；仅有 OHLC 时无法确定盘中止损和止盈的具体分钟。
    未完成交易保存在返回状态里；追加收盘线后可用同一参数重放得到更新结果。
    """
    times = long_df.index
    close_time = pd.Timestamp(close_time)
    if close_time.tz is None or (trade_from is not None and pd.Timestamp(trade_from).tz is None):
        raise ValueError("close_time 和 trade_from 需要有时区")
    if not times.equals(short_df.index):
        raise ValueError("SOXL 和 SOXS 的 K 线时间不一致")
    if len(times):
        if not isinstance(times, pd.DatetimeIndex) or times.tz is None:
            raise ValueError("K 线需要带时区的索引")
        start = close_time.tz_convert("America/New_York").normalize() + pd.Timedelta(hours=9, minutes=30)
        if not times.equals(pd.date_range(start, periods=len(times), freq=M5).tz_convert(times.tz)) or times[-1] >= close_time:
            raise ValueError("需要从常规时段开盘开始、间隔 5 分钟的连续 K 线")

    trades = []
    cash = equity = params.capital
    pnl = 0.0
    entries = 0
    halted = False
    symbol = pending = exit_reason = None
    qty = 0
    entry = raw_entry = mean_target = 0.0
    entry_i = 0
    fee = params.fee_per_order
    drag = params.fill_drag

    # 第 i 根的判断只用到 [i-lookback-1, i-2] 的基线、i-1 的偏离和 i 的确认收盘。
    if len(times):
        bars = {symbol_name: df[["open", "high", "low", "close"]].to_numpy(dtype=float)
                for symbol_name, df in (("SOXL", long_df), ("SOXS", short_df))}
        close = long_df.close
        history = close.shift(2).rolling(params.lookback)
        means = history.mean().to_numpy()
        widths = np.maximum(params.band_std * history.std(ddof=0).to_numpy(), params.min_band_pct * means)

    def sell(i, raw_exit, reason):
        nonlocal cash, pnl, symbol, qty, halted
        fill = raw_exit * (1 - drag)
        trade_pnl = qty * (fill - entry) - 2 * fee
        trades.append({
            "symbol": symbol, "qty": qty,
            "entry_t": times[entry_i].isoformat(), "exit_t": times[i].isoformat(),
            "entry": entry, "exit": fill, "reason": reason,
            "pnl": trade_pnl, "costs": qty * ((entry - raw_entry) + (raw_exit - fill)) + 2 * fee,
        })
        cash += qty * fill - fee
        pnl += trade_pnl
        halted = halted or pnl <= -params.daily_loss_pct * params.capital
        symbol, qty = None, 0

    for i, time in enumerate(times):
        final_bar = time + M5 >= close_time
        if symbol and (pending == "SELL" or final_bar):
            reason = "收盘平仓" if final_bar else exit_reason
            sell(i, float(bars[symbol][i, 0]), reason)
        elif pending and not final_bar:
            candidate = pending.split()[1]
            raw_entry = float(bars[candidate][i, 0])
            entry = raw_entry * (1 + drag)
            budget = min(params.notional, cash - 2 * fee)
            qty = max(0, math.floor(budget / entry))
            if qty:
                symbol, entry_i = candidate, i
                cash -= qty * entry + fee
                entries += 1
        pending = None

        if symbol:
            op, hi, lo, last = map(float, bars[symbol][i])
            stop = entry * (1 - params.stop_pct)
            target = entry * (1 + params.target_pct)
            # 开盘行情先于盘中波动；跳空止损按开盘，盈利跳空保守按目标价。
            if op <= stop:
                sell(i, op, "止损")
            elif op >= target:
                sell(i, target, "止盈")
            elif lo <= stop:
                sell(i, stop, "止损")
            elif hi >= target:
                sell(i, target, "止盈")

        equity = cash
        if symbol:
            equity += qty * last * (1 - drag) - fee
            if equity <= params.capital * (1 - params.daily_loss_pct):
                pending, exit_reason, halted = "SELL", "日亏损", True
            elif i - entry_i + 1 >= params.max_hold_bars:
                pending, exit_reason = "SELL", "持仓超时"
            elif ((symbol == "SOXL" and close.iat[i] >= mean_target)
                  or (symbol == "SOXS" and close.iat[i] <= mean_target)):
                pending, exit_reason = "SELL", "回归均值"
        elif (not halted and entries < params.max_trades and time + 2 * M5 < close_time
              and (trade_from is None or time >= trade_from) and i >= params.lookback + 1):
            prev, now = close.iat[i - 1], close.iat[i]
            mean, width = float(means[i]), float(widths[i])
            if prev < mean - width and prev < now < mean:
                pending, mean_target = "BUY SOXL", mean
            elif prev > mean + width and mean < now < prev:
                pending, mean_target = "BUY SOXS", mean

    return trades, {"symbol": symbol, "qty": qty, "cash": cash, "pnl": pnl, "equity": equity,
                    "pending": pending, "entry": entry if symbol else None,
                    "halted": halted, "entries": entries}
