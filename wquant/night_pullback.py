"""只使用夜盘决策时可知的数据，比较顺势回调的买入位置。"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .pair_sizing import VALIDATION_FROM
from .pair_study import CAPITAL, HISTORY_END, Entry, ideas, simulate, stats

RULES = {
    "clock": "只按方向定时买",
    "near": "均价下方 0–0.3%",
    "deep": "均价下方 0.3–1.5%",
    "both": "均价下方 0–1.5%",
    "vwap_exit": "0–1.5% 回到信号均价卖",
    "confirmed": "0–1.5% 且夜盘同向",
}


def entries(day, rule, delay=0):
    side = 1 if day.sector_return_3d >= .01 else -1 if day.sector_return_3d <= -.01 else 0
    if not side or not day.has_night:
        return []
    night = np.flatnonzero(day.L.session.eq("OVN"))
    frame = (day.L if side == 1 else day.S).iloc[night]
    typical = (frame.high + frame.low + frame.close) / 3
    vwap = (typical * frame.volume).cumsum() / frame.volume.cumsum()
    if rule == "clock":
        signals = [11, 23, 35]  # 常规撮合对应 21:00、22:00、23:00 ET
        cap = np.full(len(frame), np.inf)
    else:
        low, high = {"near": (.997, 1.), "deep": (.985, .997), "both": (.985, 1.),
                     "vwap_exit": (.985, 1.), "confirmed": (.985, 1.)}[rule]
        ratio = frame.close / vwap
        active = (ratio >= low) & ((ratio < high) if rule == "deep" else (ratio <= high))
        if rule == "confirmed":
            active &= side * (day.L.close.iloc[night] / day.prev_close - 1) > 0
        active.iloc[:11] = False
        signals = np.flatnonzero(active & ~active.shift(fill_value=False))
        cap = vwap.to_numpy() * high
    orders = []
    for signal in signals:
        fill = signal + 1 + delay
        if fill < len(night) - 1:
            end = min(fill + 12, len(night) - 1)
            orders.append(Entry(night[fill], side, night[end], stop=.015, take_pct=.015,
                                target=vwap.iat[signal] if rule == "vwap_exit" else np.nan,
                                max_buy_price=cap[signal]))
    return orders


def replay_pullback(days, rule, *, cost=1, delay=0, until_close=False, max_trades=3):
    equity, daily, trades = CAPITAL, [], []
    for day in days:
        if rule == "at_four":
            orders = [replace(e, bar=e.bar + delay) for e in ideas(day, rule)]
        else:
            orders = entries(day, rule, delay)
            if until_close:
                last = np.flatnonzero(day.L.session.eq("RTH"))[-1]
                orders = [replace(e, last=last, stop=.06, take_pct=np.inf, trail=.07) for e in orders]
        fills = simulate(day, orders, equity, cost, max_trades=max_trades)
        net = sum(t.pnl for t in fills)
        daily.append(net)
        equity += net
        trades.extend(fills)
    return daily, trades


def compare_pullbacks(days, trade_count=0):
    eligible = [d for d in days if d.has_night and np.isfinite(d.sector_return_3d)
                and d.prev_close is not None and d.prev_close > 0]
    history = [d for d in eligible if d.date <= HISTORY_END]
    late = [d for d in history if d.date >= VALIDATION_FROM]
    early = [d for d in history if d.date < VALIDATION_FROM]
    fresh = [d for d in eligible if d.date > HISTORY_END]
    if not early or not late:
        raise ValueError("夜盘、昨收及行业趋势的共同样本未覆盖两个历史区间")
    print(f"共同样本 {history[0].date} 至 {history[-1].date}，{len(history)} 晚，"
          f"偏多 {sum(d.sector_return_3d >= .01 for d in history)} 晚，"
          f"偏空 {sum(d.sector_return_3d <= -.01 for d in history)} 晚。"
          f"收益率以各区间起始本金 ${CAPITAL:,.0f} 计算，每笔最多 $1,000。")
    periods = [("全期", history, 1, 0), ("前段", early, 1, 0), ("后段", late, 1, 0),
               ("后段延迟5分钟及双倍成本", late, 2, 1)]
    if fresh:
        periods.append(("新增", fresh, 1, 0))
    for label, sample, cost, delay in periods:
        print(f"\n{label} {len(sample)} 晚\n规则  笔数  净利($)  本金收益率  胜率  日末最大回撤")
        for rule, name in RULES.items():
            daily, trades = replay_pullback(sample, rule, cost=cost, delay=delay)
            metrics = stats((daily, trades))
            n, net = metrics["n"], metrics["net"]
            wins = f"{metrics['wins']/n:.1%}" if n else "—"
            print(f"{name}  {n}  {net:+,.2f}  {net/CAPITAL:+.2%}  {wins}  {metrics['dd']:.2%}")
            if rule == "both" and label == "全期":
                for sym in ("SOXL", "SOXS"):
                    leg = [t for t in trades if t.symbol == sym]
                    pnl = sum(t.pnl for t in leg)
                    print(f"  {sym} {len(leg)} 笔 {pnl:+.2f} 美元，本金收益贡献 {pnl/CAPITAL:+.2%}")
                if trade_count > 0:
                    for t in trades[-trade_count:]:
                        print(f"  {t.symbol} {t.qty} 股 {t.entry_time} → {t.exit_time} "
                              f"{t.reason} {t.pnl:+.2f} 美元")
    if not fresh:
        print("新增样本中完整且有方向参考的数据为 0 晚，新增收益不可计算。")
    # 昼夜持仓要求整个盘前也有可模拟止损的连续行情。
    print("\n同一批夜盘与日盘完整日，每晚最多一笔；04:00 对照使用原方向规则。")
    for label, sample, cost, delay in periods:
        complete = [d for d in sample if d.has_pre and d.night_signal_close is not None]
        print(f"{label} {len(complete)} 晚")
        if not complete:
            continue
        for rule, long_hold in (("both", False), ("both", True), ("confirmed", False),
                                ("confirmed", True), ("at_four", True)):
            daily, trades = replay_pullback(complete, rule, cost=cost, delay=delay,
                                           until_close=long_hold, max_trades=1)
            metrics = stats((daily, trades))
            name = RULES.get(rule, "04:00 方向对照")
            style = "6%初始止损/7%移动止损/收盘前退出" if long_hold else "1小时止盈止损"
            print(f"{name} {style} {len(trades)} 笔 {metrics['net']:+.2f} 美元"
                  f"（{metrics['net']/CAPITAL:+.2%}），日末最大回撤 {metrics['dd']:.2%}")
