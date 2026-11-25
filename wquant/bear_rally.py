"""弱市夜盘反弹中买 SOXS，观察次日常规盘开盘的反转。"""
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import data, intraday_data
from .pair_data import _close, _merge, _read, _sector_features, _segment
from .pair_study import CAPITAL, Entry, simulate, stats
from .scalp_data import ET, M5, session_bounds


@dataclass
class Morning:
    date: date
    trend: float
    L: pd.DataFrame
    S: pd.DataFrame
    complete: bool


def market_sources(forward):
    left, right = [intraday_data.load(s, adjust=False) for s in ("SOXL", "SOXS")]
    sector = _close(_read(data.CACHE / "SOXX.csv", daily_zone=ET))
    sources = [(left, right, sector, None)]
    if forward:
        paths = [forward / f"{s}_M5.csv" for s in ("SOXL", "SOXS")]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        newer = [_read(path) for path in paths]
        sector_file = forward / "SOXX_1d.csv"
        if not sector_file.exists():
            sector_file = forward / "SOXX_daily.csv"
        sector = _merge(sector, _close(_read(sector_file, daily_zone=ET)))
        cutoff = max(left.index[left.session == "RTH"].date)
        sources.append((*newer, sector, cutoff))
    return sources


def mornings(forward):
    result = []
    for left, right, sector, cutoff in market_sources(forward):
        _, moves = _sector_features(sector)
        for day in sorted(set(left.index[left.session == "RTH"].date)):
            bounds = session_bounds(day)
            if not bounds or (cutoff and day <= cutoff):
                continue
            opening = bounds[0]
            rth = _segment(left, right, opening, opening + 13 * M5, "RTH")
            if rth is None:
                continue
            prev = day - timedelta(days=1)
            while not session_bounds(prev):
                prev -= timedelta(days=1)
            night_start = pd.Timestamp(day - timedelta(days=1), tz=ET).replace(hour=20)
            pre = opening.replace(hour=4, minute=0)
            night = _segment(left, right, night_start, pre, "OVN")
            pm = _segment(left, right, pre, opening, "PRE")
            complete = night is not None and pm is not None
            l, s = (map(pd.concat, zip(night, pm, rth)) if complete else rth)
            result.append(Morning(day, moves.get(pd.Timestamp(prev), np.nan), l, s, complete))
    return result


def rally_orders(day, delay=0, *, min_ratio=.985, min_rally=.01, side=-1, require_market_move=True):
    selected = day.S if side == -1 else day.L
    night = selected.loc[selected.session == "OVN"]
    bull = day.L.loc[night.index]
    typical = (night.high + night.low + night.close) / 3
    vwap = (typical * night.volume).cumsum() / night.volume.cumsum()
    anchor = bull.open.iat[0]
    rally = ((bull.close > anchor) & (bull.close >= (1 + min_rally) * anchor) if side == -1
             else (bull.close < anchor) & (bull.close <= (1 - min_rally) * anchor))
    discount = night.close / vwap
    active = discount.ge(min_ratio) & discount.lt(1)
    if require_market_move:
        active &= rally
    active.iloc[:11] = False
    triggers = np.flatnonzero(active & ~active.shift(fill_value=False))
    return [Entry(i + 1 + delay, side, len(night), stop=.06, take_pct=np.inf,
                  max_buy_price=vwap.iat[i])
            for i in triggers if i + 1 + delay < len(night)]


def replay(days, minutes, *, delay=0, cost=1, min_ratio=.985, min_rally=.01, stop=.06):
    daily, fills = [], []
    equity = CAPITAL
    for d in days:
        opening = np.flatnonzero(d.S.session.eq("RTH"))[0]
        orders = [replace(e, last=opening + minutes // 5, stop=stop) for e in
                  rally_orders(d, delay, min_ratio=min_ratio, min_rally=min_rally)]
        trades = simulate(d, orders, equity, cost, max_trades=1)
        pnl = sum(t.pnl for t in trades)
        daily.append(pnl)
        fills.extend(trades)
        equity += pnl
    return daily, fills


def open_replay(days, delay=0, cost=1):
    equity, daily, fills = CAPITAL, [], []
    for d in days:
        opening = np.flatnonzero(d.S.session.eq("RTH"))[0]
        entry = Entry(opening + delay, -1, opening + 6, stop=.06, take_pct=np.inf)
        trades = simulate(d, [entry], equity, cost)
        pnl = sum(t.pnl for t in trades)
        equity += pnl
        daily.append(pnl)
        fills.extend(trades)
    return stats((daily, fills))


def compare_rallies(forward, trade_count=0):
    days = mornings(forward)
    eligible = [d for d in days if d.complete and np.isfinite(d.trend)]
    if not eligible:
        raise ValueError("没有完整夜盘至开盘后一小时且具有行业趋势记录的交易日")
    bear = [d for d in eligible if d.trend <= -.01]
    other = [d for d in eligible if d.trend > -.01]
    month = max(d.date for d in days).replace(day=1)
    recent = [d for d in days if d.date >= month]
    groups = [("全期偏空", bear), ("全期其余趋势", other),
              ("当月前偏空", [d for d in bear if d.date < month]),
              ("当月偏空", [d for d in bear if d.date >= month])]
    for label, sample in groups:
        if not sample:
            print(label, "无合格样本")
            continue
        for minute, delay, cost in ((0,0,1), (30,0,1), (60,0,1), (30,1,2)):
            daily, trades = replay(sample, minute, delay=delay, cost=cost)
            m = stats((daily,trades))
            print(f"{label} {len(sample)} 晚，开盘后 {minute} 分钟退出，延迟 {delay*5} 分钟，"
                  f"成本 {cost} 倍，{len(trades)} 笔，盈利 {m['wins']} 笔，净利 {m['net']:+.2f} 美元"
                  f"（{m['net']/CAPITAL:+.2%}），日末回撤 {m['dd']:.2%}")
    _, trades = replay(bear, 30)
    by_day = {t.day:t for t in trades}
    matched = [d for d in bear if d.date in by_day]
    falls = 0
    stopped_before_open = 0
    for d in matched:
        opening = np.flatnonzero(d.S.session.eq("RTH"))[0]
        falls += int(d.L.open.iat[opening + 6] < d.L.open.iat[opening])
        t = by_day[d.date]
        stopped_before_open += int(t.exit_time < d.S.index[opening])
    print(f"夜盘成交的 {len(matched)} 天，SOXL 开盘半小时收低 {falls} 天，"
          f"夜盘仓位在开盘前触发止损 {stopped_before_open} 天。")
    for label, sample in (("全期", matched), ("当月前", [d for d in matched if d.date < month]),
                          ("当月", [d for d in matched if d.date >= month])):
        for delay, cost in ((0, 1), (1, 2)):
            m = open_replay(sample, delay, cost)
            print(f"{label}同日等到开盘后 {delay*5} 分钟买 SOXS，最迟 10:00 卖，成本 {cost} 倍，"
                  f"{m['n']} 笔，{m['wins']} 胜，净利 {m['net']:+.2f} 美元（{m['net']/CAPITAL:+.2%}）")
    wide = stats(replay(bear, 30, min_ratio=0))
    held = stats(replay(bear, 30, stop=1.))
    print(f"相同夜盘买点一路持有至 10:00 的价格对照，{held['n']} 笔，"
          f"扣费净利 {held['net']:+.2f} 美元（{held['net']/CAPITAL:+.2%}）。")
    print(f"只要求低于 VWAP、取消最大折价限制，夜盘买 SOXS 最迟 10:00 卖，{wide['n']} 笔，"
          f"净利 {wide['net']:+.2f} 美元（{wide['net']/CAPITAL:+.2%}）。")
    broad_recent = {}
    for label, sample in (("全期偏空", bear), ("其余趋势", other),
                          ("当月偏空", [d for d in bear if d.date >= month])):
        if sample:
            base = replay(sample, 30, min_ratio=0, min_rally=0)
            m = stats(base)
            stress = stats(replay(sample, 30, min_ratio=0, min_rally=0, delay=1, cost=2))
            if label == "当月偏空":
                broad_recent = {t.day:t for t in base[1]}
            print(f"{label}宽松条件（SOXL 高于夜盘开盘且 SOXS 低于 VWAP），夜盘买、最迟 10:00 卖，"
                  f"{m['n']} 笔，{m['wins']} 胜，净利 {m['net']:+.2f} 美元（{m['net']/CAPITAL:+.2%}），"
                  f"日末回撤 {m['dd']:.2%}，延迟 5 分钟及双倍成本 {stress['n']} 笔，"
                  f"{stress['net']:+.2f} 美元（{stress['net']/CAPITAL:+.2%}）。")
    unmatched = open_replay([d for d in bear if d.date not in by_day])
    print(f"同为偏空、夜盘未成交的日期，09:30 买 SOXS 最迟 10:00 卖，{unmatched['n']} 笔，"
          f"{unmatched['wins']} 胜，净利 {unmatched['net']:+.2f} 美元"
          f"（{unmatched['net']/CAPITAL:+.2%}）。")
    print("近期逐日，美东常规交易日 / SOXX 前三日 / SOXL 开盘后30分钟 / 1%拉高规则 / 宽松规则")
    for d in recent:
        rth = d.L.loc[d.L.session == "RTH"]
        move = rth.open.iat[6] / rth.open.iat[0] - 1
        t = by_day.get(d.date)
        status = ("夜盘至开盘路径缺失" if not d.complete else "趋势缺失" if not np.isfinite(d.trend)
                  else "未达偏空阈值" if d.trend > -.01 else "未成交")
        print(d.date, f"{d.trend:+.2%}", f"{move:+.2%}",
              f"{t.entry_time:%H:%M} 买入 {t.pnl:+.2f} 美元（{t.pnl/CAPITAL:+.2%}）" if t else status,
              f"{broad_recent[d.date].pnl:+.2f} 美元" if d.date in broad_recent else status)
    if trade_count > 0:
        for t in trades[-trade_count:]:
            print(t.entry_time, t.exit_time, t.reason, f"{t.pnl:+.2f} 美元")
