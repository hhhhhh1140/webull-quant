"""固定时刻方向策略的现金仓位对照。仅用于模拟回放。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .entry_clock import clock_entry
from .pair_study import CAPITAL, FEE, HISTORY_END, Entry, eligible, ideas, simulate
from .pair_data import Day
from .sizing_news import NewsBook, NewsDecision

VALIDATION_FROM = date(2026, 9, 18)  # 与原策略比较使用同一后段起始日
BASE = 1 / 3
LABELS = {"base": "固定三分之一", "full": "每次全仓", "strong": "强信号全仓",
          "events": "强信号＋事件限仓", "news": "强信号＋事件＋新闻确认", "matched": "等平均仓位对照"}
HERE = Path(__file__).parent


@dataclass
class Signal:
    day: Day
    order: Entry
    move: float
    korea: float
    news: NewsDecision
    earnings: tuple[str, ...]

    @property
    def strong(self):
        return abs(self.move) >= .06 and self.order.side * self.korea >= .01


def load_signals(days, book, earnings_path, *, clock="04:00"):
    earnings = pd.read_csv(earnings_path)
    releases = [(pd.Timestamp(r.published_at), r.issuer) for r in earnings.itertuples()]
    if any(t.tz is None for t, _ in releases):
        raise ValueError("财报发布时间缺少时区")
    result = []
    for d in days:
        if clock == "04:00":
            entries = ideas(d, "at_four")
            if not entries:
                continue
            e = entries[0]
            signal_close, korea = d.night_signal_close, d.korea_at_four
        else:
            e = clock_entry(d, clock)
            if e is None:
                continue
            prior_bar = d.L.index[e.bar] - pd.Timedelta(minutes=5)
            signal_close, korea = d.L.close.at[prior_bar], d.korea_move.at[prior_bar]
        cutoff = d.L.index[e.bar]
        end = d.L.index[e.last]
        released = tuple(s for t, s in releases if cutoff - pd.Timedelta(hours=12) <= t <= cutoff)
        result.append(Signal(d, e, signal_close / d.prev_close - 1, korea,
                             book.for_session(cutoff, end), released))
    return result


def allocation(signal, policy, matched=BASE):
    if policy == "matched":
        return matched
    if policy == "full":
        return 1.
    if policy == "base":
        return BASE
    if policy not in ("strong", "events", "news"):
        raise ValueError(f"未知仓位规则 {policy}")
    if policy == "news":
        if signal.news.contradicts(signal.order.side) or signal.news.coverage == "conflict":
            return 0.
        if not signal.news.confirms(signal.order.side):
            return BASE
    if policy != "strong" and (signal.news.scheduled_risk or signal.earnings):
        return BASE
    return 1. if signal.strong else BASE


def replay_sizing(signals, policy, matched=BASE, *, delay=0, cost=1):
    equity = CAPITAL
    marks = [equity]
    records = []
    for s in signals:
        weight = allocation(s, policy, matched)
        entry = replace(s.order, bar=s.order.bar + delay)
        fills = simulate(s.day, [entry], equity, cost, notional=weight * equity)
        pnl = 0.
        if fills:
            t, = fills
            pnl = t.pnl
            bars = s.day.L if entry.side == 1 else s.day.S
            held = bars.loc[(bars.index >= t.entry_time) & (bars.index < t.exit_time)]
            sell_drag = np.where(held.session.eq("RTH"), .0005, .0015) * cost
            cash = equity - t.qty * t.entry_price - FEE * cost
            marks.extend(cash + t.qty * held.close.to_numpy() * (1 - sell_drag) - FEE * cost)
        records.append({"day": s.day.date, "full": weight == 1 and bool(fills), "weight": weight,
                        "pnl": pnl, "return": pnl / equity, "filled": bool(fills)})
        equity += pnl
        marks.append(equity)
    values = np.array(marks)
    drawdown = float((values / np.maximum.accumulate(values) - 1).min())
    return records, drawdown


def compare_sizing(days, news_path=None, trade_count=0):
    book = NewsBook.load(news_path, HERE / "sizing_events.csv")
    available = [d for d in days if eligible(d, "at_four")]
    history = [d for d in available if d.date <= HISTORY_END]
    history_signals = load_signals(history, book, HERE / "sizing_earnings.csv")
    train = [s for s in history_signals if s.day.date < VALIDATION_FROM]
    later = [s for s in history_signals if s.day.date >= VALIDATION_FROM]
    if not train or not later:
        raise ValueError("选择期或后段缺少可用 04:00 交易信号")
    matched = float(np.mean([allocation(s, "events") for s in train]))
    print(f"04:00 仓位研究  {history[0].date} 至 {history[-1].date}，可回放 {len(history)} 天，"
          f"信号 {len(history_signals)} 次，其中后段 {len(later)} 次。")
    print(f"先前区间的事件限仓策略平均目标仓位 {matched:.1%}，以此固定仓位作对照。"
          "每种策略和单独检验的后段均从 3,000 美元起算。")
    rows = []
    for policy, name in LABELS.items():
        records, dd = replay_sizing(history_signals, policy, matched)
        tail, _ = replay_sizing(later, policy, matched)
        stress, _ = replay_sizing(later, policy, matched, delay=1, cost=2)
        rows.append({"规则": name, "全仓笔数": sum(r["full"] for r in records),
                     "净利$": sum(r["pnl"] for r in records), "5分钟回撤": f"{dd:.1%}",
                     "最差日": f"{min(r['return'] for r in records):.1%}",
                     "后段净利$": sum(r["pnl"] for r in tail), "后段压力净利$": sum(r["pnl"] for r in stress)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    strong = [s for s in history_signals if s.strong]
    event_days = [s for s in history_signals if s.earnings or s.news.scheduled_risk]
    print(f"强信号 {len(strong)} 次，遇事件的强信号 {sum(s.strong for s in event_days)} 次。"
          f"全部信号中，宏观日 {sum(s.news.scheduled_risk for s in history_signals)} 次，财报夜 "
          f"{sum(bool(s.earnings) for s in history_signals)} 次，收到已标注新闻 "
          f"{sum(s.news.headlines > 0 for s in history_signals)} 次。")
    for label, group in (("事件限仓后可全仓", [s for s in strong if not (s.earnings or s.news.scheduled_risk)]),
                         ("其余信号", [s for s in history_signals if allocation(s, "events") < 1])):
        # 所有机会用相同的现金仓位比较交易回报，去除因仓位大小产生的收益差异。
        records, _ = replay_sizing(group, "full")
        if records:
            print(f"{label} {len(records)} 笔，按相同全仓口径平均单次净回报 "
                  f"{np.mean([r['return'] for r in records]):.2%}，"
                  f"盈利 {sum(r['pnl'] > 0 for r in records)} 笔。")
    fresh = [d for d in available if d.date > HISTORY_END]
    fresh_signals = load_signals(fresh, book, HERE / "sizing_earnings.csv")
    print(f"{HISTORY_END} 之后，可回放 {len(fresh)} 天、{len(fresh_signals)} 次信号。")
    if fresh_signals:
        for policy in ("base", "events", "news"):
            records, _ = replay_sizing(fresh_signals, policy)
            print(f"新增区间 {LABELS[policy]} 净利 ${sum(r['pnl'] for r in records):.2f}")
    if trade_count > 0:
        for s in (history_signals + fresh_signals)[-trade_count:]:
            print(f"{s.day.date} {'SOXL' if s.order.side == 1 else 'SOXS'} 夜盘 {s.move:+.1%} "
                  f"KOSPI {s.korea:+.1%} 宏观事件 {s.news.scheduled_risk} 财报 {','.join(s.earnings) or '—'} "
                  f"消息 {s.news.coverage} 事件限仓 {allocation(s,'events'):.0%} "
                  f"新闻确认仓位 {allocation(s,'news'):.0%}")
