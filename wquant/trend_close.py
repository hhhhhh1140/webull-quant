"""按上个收盘确定方向，夜盘买回调，持有到下一常规盘的最后五分钟线。"""
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .bear_rally import market_sources, rally_orders
from .pair_data import _sector_features, _segment
from .pair_sizing import VALIDATION_FROM
from .pair_study import CAPITAL, FEE, simulate, stats
from .scalp_data import ET, M5, OHLCV, session_bounds

DEFAULT_THRESHOLD = .05


def trend_side(move, threshold):
    move = round(move, 12)  # 收盘价比值计算的舍入误差不应排除恰好达到门槛的交易日。
    return 1 if move >= threshold else -1 if move <= -threshold else 0


@dataclass
class CloseDay:
    date: date
    trend: float
    L: pd.DataFrame
    S: pd.DataFrame
    missing_marks: int


def valid_marks(frame, start, opening, end):
    part = frame.loc[start:end - M5]
    labels = np.where(part.index < opening, "PRE", "RTH")
    valid = ((part.session == labels) & (part[OHLCV] > 0).all(axis=1)
             & np.isfinite(part[OHLCV]).all(axis=1)
             & (part.low <= part[["open", "close"]].min(axis=1))
             & (part.high >= part[["open", "close"]].max(axis=1)))
    return part.loc[valid & ~part.index.duplicated(keep=False)]


def close_days(forward, *, exit_bars_before=1):
    result = []
    for left, right, sector, cutoff in market_sources(forward):
        _, moves = _sector_features(sector)
        dates = sorted(set(left.index[left.session == "RTH"].date))
        for day in dates:
            bounds = session_bounds(day)
            if not bounds or (cutoff and day <= cutoff):
                continue
            opening, close = bounds
            end = close - (exit_bars_before - 1) * M5
            prev = day - timedelta(days=1)
            while not session_bounds(prev):
                prev -= timedelta(days=1)
            trend = moves.get(pd.Timestamp(prev), np.nan)
            night_start = pd.Timestamp(day - timedelta(days=1), tz=ET).replace(hour=20)
            pre = opening.replace(hour=4, minute=0)
            night = _segment(left, right, night_start, pre, "OVN")
            if night is None or not np.isfinite(trend):
                continue
            # 中途行情只用来估值。实际入场夜盘和尾盘成交线必须可核验。
            l, s = (valid_marks(frame, pre, opening, end) for frame in (left, right))
            grid = pd.date_range(pre, end, freq=M5, inclusive="left")
            common = grid.intersection(l.index).intersection(s.index)
            if end - M5 not in common:
                continue
            result.append(CloseDay(day, trend, pd.concat([night[0], l.loc[common]]),
                                   pd.concat([night[1], s.loc[common]]), len(grid) - len(common)))
    return result


def replay_close(days, delay=0, cost=1, *, require_market_move=True, threshold=DEFAULT_THRESHOLD):
    equity, daily, trades, marks = CAPITAL, [], [], [CAPITAL]
    for d in days:
        side = trend_side(d.trend, threshold)
        signals = (rally_orders(d, delay, min_ratio=0, min_rally=0, side=side,
                                require_market_move=require_market_move) if side else [])
        orders = [replace(e, last=len(d.L) - 1, stop=1., take_pct=np.inf, trail=0.) for e in signals]
        fills = simulate(d, orders, equity, cost, max_trades=1)
        net = 0.
        if fills:
            t, = fills
            bars = d.L if side == 1 else d.S
            held = bars.loc[(bars.index >= t.entry_time) & (bars.index < t.exit_time)]
            drag = np.where(held.session.eq("RTH"), .0005, .0015) * cost
            cash = equity - t.qty * t.entry_price - 2 * FEE * cost
            marks.extend(cash + t.qty * held.close.to_numpy() * (1 - drag))
            trades.append(t)
            net = t.pnl
        equity += net
        daily.append(net)
        marks.append(equity)
    values = np.array(marks)
    return daily, trades, float((values / np.maximum.accumulate(values) - 1).min())


def compare_thresholds(days):
    recent = days[-1].date.replace(day=1)
    print("仅按趋势与 ETF 自身 VWAP 夜盘入场，各门槛使用同一批日期，收益率分母均为初始本金。")
    for threshold in (.01, .03, .05):
        daily, trades, dd = replay_close(days, threshold=threshold, require_market_move=False)
        stress_daily, stress_trades, _ = replay_close(days, delay=1, cost=2, threshold=threshold,
                                                      require_market_move=False)
        metrics = stats((daily, trades))
        print(f"±{threshold:.0%}：{len(trades)} 笔，{metrics['wins']} 胜，"
              f"净利 {metrics['net']:+.2f} 美元（{metrics['net']/CAPITAL:+.2%}），"
              f"五分钟回撤 {dd:.2%}，压力测试 {len(stress_trades)} 笔，"
              f"{sum(stress_daily):+.2f} 美元（{sum(stress_daily)/CAPITAL:+.2%}）")
        for symbol in ("SOXL", "SOXS"):
            leg = [t for t in trades if t.symbol == symbol]
            net = sum(t.pnl for t in leg)
            print(f"  {symbol} {len(leg)} 笔，{net:+.2f} 美元（{net/CAPITAL:+.2%}）")
        for label, start, end in (("前段", days[0].date, VALIDATION_FROM),
                                  ("后段", VALIDATION_FROM, date.max),
                                  ("最近可回放月", recent, date.max)):
            subset = [t for t in trades if start <= t.day < end]
            net = sum(t.pnl for t in subset)
            print(f"  {label} {len(subset)} 笔，{net:+.2f} 美元（{net/CAPITAL:+.2%}）")
    # 额外观察提高上涨门槛、保留下跌门槛时的表现。
    asymmetric = [d for d in days if trend_side(d.trend, .05) == 1 or trend_side(d.trend, .03) == -1]
    for delay, cost in ((0, 1), (1, 2)):
        daily, trades, dd = replay_close(asymmetric, delay=delay, cost=cost, threshold=.03,
                                         require_market_move=False)
        net = sum(daily)
        print(f"探索对照 +5% / −3%，延迟 {delay*5} 分钟、成本 {cost} 倍，{len(trades)} 笔，"
              f"净利 {net:+.2f} 美元（{net/CAPITAL:+.2%}），五分钟回撤 {dd:.2%}")


def compare_close(forward, trade_count=0):
    days = close_days(forward)
    if not days:
        raise ValueError("没有可验证的完整夜盘、行业趋势和尾盘成交价")
    recent = days[-1].date.replace(day=1)
    print(f"{days[0].date} 至 {days[-1].date}，{len(days)} 个可回放交易日，"
          f"{sum(d.missing_marks > 0 for d in days)} 日持仓估值行情有缺口。"
          "模拟本金 3,000 美元，每笔最多 1,000 美元。")
    compare_thresholds(days)
    print(f"按 ±{DEFAULT_THRESHOLD:.0%} 定义趋势的详细回放，先列出附加夜盘方向条件的结果")
    periods = (("全期", days), ("上涨", [d for d in days if trend_side(d.trend, DEFAULT_THRESHOLD) == 1]),
               ("下跌", [d for d in days if trend_side(d.trend, DEFAULT_THRESHOLD) == -1]),
               ("前段", [d for d in days if d.date < VALIDATION_FROM]),
               ("后段", [d for d in days if d.date >= VALIDATION_FROM]),
               ("最近可回放月", [d for d in days if d.date >= recent]))
    for label, sample in periods:
        if not sample:
            continue
        daily, trades, dd = replay_close(sample)
        m = stats((daily, trades))
        print(f"{label} {len(sample)} 天，{m['n']} 笔，{m['wins']} 胜，"
              f"净利 {m['net']:+.2f} 美元（{m['net']/CAPITAL:+.2%}），"
              f"日末回撤 {m['dd']:.2%}，五分钟估值回撤 {dd:.2%}")
        if label == "最近可回放月" or (label == "全期" and trade_count > 0):
            for t in (trades if label == "最近可回放月" else trades[-trade_count:]):
                print(f"  {t.symbol} {t.entry_time} → {t.exit_time}  {t.pnl:+.2f} 美元（{t.pnl/CAPITAL:+.2%}）")
    daily, trades, _ = replay_close(days, delay=1, cost=2)
    print(f"全期延迟五分钟买入、成交成本翻倍，{len(trades)} 笔，"
          f"{sum(daily):+.2f} 美元（{sum(daily)/CAPITAL:+.2%}）")
    for label, sample in periods:
        if not sample:
            continue
        daily, trades, dd = replay_close(sample, require_market_move=False)
        print(f"{label} 只要求趋势方向和所选 ETF 低于 VWAP，{len(trades)} 笔，"
              f"{sum(daily):+.2f} 美元（{sum(daily)/CAPITAL:+.2%}），五分钟估值回撤 {dd:.2%}")
    daily, trades, _ = replay_close(days, delay=1, cost=2, require_market_move=False)
    print(f"只按趋势和 VWAP，晚五分钟买入、双倍成本，{len(trades)} 笔，"
          f"{sum(daily):+.2f} 美元（{sum(daily)/CAPITAL:+.2%}）")
    earlier = close_days(forward, exit_bars_before=2)
    shared = {d.date for d in days} & {d.date for d in earlier}
    for label, sample in (("收盘前五分钟（共同日期）", [d for d in days if d.date in shared]),
                          ("收盘前十分钟（共同日期）", [d for d in earlier if d.date in shared]),
                          ("最近月收盘前十分钟", [d for d in earlier if d.date.replace(day=1) == recent])):
        for required in (False, True):
            daily, trades, _ = replay_close(sample, require_market_move=required)
            entry_rule = "加夜盘方向条件" if required else "只按趋势和 VWAP"
            print(f"{label} {entry_rule} {len(sample)} 天，{len(trades)} 笔，"
                  f"{sum(daily):+.2f} 美元（{sum(daily)/CAPITAL:+.2%}）")
