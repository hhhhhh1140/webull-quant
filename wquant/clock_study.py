"""在相同交易日比较固定进场时刻，单独检验推迟成交以及条件仓位。"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .entry_clock import clock_entry, eligible_clock
from .pair_sizing import HERE, VALIDATION_FROM, load_signals, replay_sizing
from .pair_study import CAPITAL, HISTORY_END, simulate, stats
from .sizing_news import NewsBook


# 美东时间，从前一自然日晚间起排序；在查看本轮报酬前冻结候选。
CLOCKS = ("21:00", "23:00", "01:30", "03:00", "03:30", "03:55", "04:00", "04:05",
          "04:30", "05:00", "07:00", "09:00", "09:30", "09:35", "10:00", "12:00", "14:00")
REFERENCE = "04:00"


def replay_clock(days, clock, *, cost=1, delay=0, defer_to=None):
    """defer_to 只改成交时间，当天是否交易和方向均在 clock 时确定。"""
    equity = CAPITAL
    daily, trades = [], []
    for day in days:
        entry = clock_entry(day, clock, delay=delay)
        if entry is not None and defer_to:
            hour, minute = map(int, defer_to.split(":"))
            idx = day.L.index
            slots = ((idx.hour == hour) & (idx.minute == minute)).nonzero()[0]
            if len(slots) != 1 or not entry.bar <= slots[0] < entry.last:
                raise ValueError("推迟成交必须在信号确认后、当日平仓前")
            entry = replace(entry, bar=int(slots[0]))
        fills = simulate(day, [entry], equity, cost) if entry is not None else []
        net = sum(t.pnl for t in fills)
        equity += net
        daily.append(net)
        trades.extend(fills)
    return daily, trades


def compare_clocks(days, news_path=None, trade_count=0):
    common = [d for d in days if d.date <= HISTORY_END and all(eligible_clock(d, c, delay=1) for c in CLOCKS)]
    early = [d for d in common if d.date < VALIDATION_FROM]
    late = [d for d in common if d.date >= VALIDATION_FROM]
    if not early or not late:
        raise ValueError("进场时点的共同完整行情没有覆盖前段和后段")
    fresh = [d for d in days if d.date > HISTORY_END]
    print(f"共同样本 {len(common)} 日，{common[0].date} 至 {common[-1].date}。前段 {len(early)} 日，"
          f"自 {VALIDATION_FROM} 起复核 {len(late)} 日。每笔最多 $1,000，每日最多一次。")
    rows, late_pnl = [], {}
    ranked = {}
    for clock in CLOCKS:
        base = stats(replay_clock(common, clock))
        train_stress = stats(replay_clock(early, clock, cost=2, delay=1))
        validation = replay_clock(late, clock)
        late_pnl[clock] = validation[0]
        late_net = sum(validation[0])
        validation_stress = stats(replay_clock(late, clock, cost=2, delay=1))
        if clock != REFERENCE and train_stress["n"] >= 10 and train_stress["net"] > 0:
            ranked[clock] = train_stress["net"]
        available = [d for d in fresh if eligible_clock(d, clock)]
        newest = stats(replay_clock(available, clock)) if available else None
        rows.append({"美东": clock, "全期笔数": base["n"], "全期净利$": base["net"],
                     "日末回撤": f"{base['dd']:.1%}", "前段压力$": train_stress["net"],
                     "后段笔数": len(validation[1]), "后段净利$": late_net,
                     "后段压力$": validation_stress["net"], "新增天数": len(available),
                     "新增净利$": newest["net"] if newest else None})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, na_rep="—", float_format=lambda n: f"{n:,.2f}"))
    print("压力列按原信号晚 5 分钟成交、点差滑点和费用翻倍。前一晚的时点按下一常规交易日记账。")
    selected = [REFERENCE]
    if ranked:
        best_alt = max(ranked, key=ranked.get)
        selected.append(best_alt)
        diff = [x-y for x,y in zip(late_pnl[best_alt], late_pnl[REFERENCE])]
        print(f"按前段压力净利及至少 10 笔的门槛选出的备选为 {best_alt}。"
              f"后段相对 04:00 的逐日净利差合计 ${sum(diff):.2f}，"
              f"去掉对备选最有利的一天为 ${sum(diff)-max(diff):.2f}。")
        first, second = sorted(selected, key=CLOCKS.index)
        for period, sample in (("全期", common), ("后段", late)):
            immediate = stats(replay_clock(sample, first))
            deferred = stats(replay_clock(sample, first, defer_to=second))
            print(f"{period} 固定 {first} 的信号和方向，{immediate['n']} 笔，"
                  f"{first} 买入 ${immediate['net']:.2f}，等到 {second} 买入 ${deferred['net']:.2f}。")
    else:
        print("前段没有其他时点同时满足压力净利为正、至少 10 笔交易。")
    later_clocks = [clock for clock in CLOCKS[CLOCKS.index(REFERENCE) + 1:] if clock in ranked]
    if later_clocks:
        later_choice = max(later_clocks, key=ranked.get)
        print(f"只看 04:00 之后的候选，同一前段门槛和排名选出 {later_choice}。")
        if later_choice not in selected:
            selected.append(later_choice)
    leader = max(CLOCKS, key=lambda clock: sum(late_pnl[clock]))
    if leader != REFERENCE:
        diffs = pd.Series(late_pnl[leader], index=[d.date for d in late]) - late_pnl[REFERENCE]
        print(f"事后查看，{leader} 的后段净利最高，较 04:00 多 ${diffs.sum():.2f}。"
              f"剔除差额最大的一天 {diffs.idxmax()} 后，差额为 ${diffs.sum()-diffs.max():.2f}。")
    book = NewsBook.load(news_path, HERE / "sizing_events.csv")
    for clock in selected:
        signals = load_signals(late, book, HERE / "sizing_earnings.csv", clock=clock)
        for policy in (("events", "news") if news_path else ("events",)):
            records, dd = replay_sizing(signals, policy)
            stress, _ = replay_sizing(signals, policy, delay=1, cost=2)
            print(f"{clock} 后段{'新闻确认' if policy == 'news' else '强信号加事件限仓'}  "
                  f"全仓 {sum(r['full'] for r in records)} 次，净利 ${sum(r['pnl'] for r in records):.2f}，"
                  f"五分钟权益回撤 {dd:.1%}，压力净利 ${sum(r['pnl'] for r in stress):.2f}。")
        if trade_count > 0:
            for t in replay_clock(common, clock)[1][-trade_count:]:
                print(f"{clock} {t.symbol} {t.qty} 股  {t.entry_time} → {t.exit_time}  净利 ${t.pnl:.2f}")
    return table
