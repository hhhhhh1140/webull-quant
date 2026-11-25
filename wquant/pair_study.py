"""SOXL/SOXS 分时策略对照实验。开平仓均为模拟成交，金额单位为美元。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd

from .pair_signals import RTH_RULES, signals


HISTORY_END = date(2026, 11, 6)  # 此日期之后的数据不参与制定规则或排名
CAPITAL = 3000.0
NOTIONAL = 1000.0
FEE = 0.10
RULES = (*RTH_RULES, "night_vwap", "night_trend", "night_korea", "at_four", "pre_gap")
CONTROLS = ("cash", "hold_long", "hold_short")
LABELS = dict(zip((*RULES, *CONTROLS), (
    "开盘区间突破", "盘中趋势回踩", "盘中VWAP回归", "弱市夜盘SOXS回调", "夜盘动量",
    "韩国确认夜盘动量", "04:00方向", "盘前方向常规盘进场", "空仓", "开盘买SOXL", "开盘买SOXS",
)))


@dataclass(frozen=True)
class Entry:
    bar: int
    side: int
    last: int
    stop: float = 0.03
    target: float = np.nan  # 已知价格，NaN 表示使用相对目标
    take_pct: float = 0.06
    trail: float = 0.0
    max_buy_price: float = np.inf  # 仅在模拟买价（含滑点）不高于上限时成交


def eligible(day, rule):
    if rule == "night_vwap":
        return day.has_night and pd.notna(day.bear)
    if rule == "at_four":
        return day.has_pre and pd.notna(day.night_signal_close) and pd.notna(day.prev_close)
    if rule in ("night_trend", "night_korea"):
        return (day.has_pre and day.has_night and pd.notna(day.prev_close)
                and (rule != "night_korea" or day.has_korea_history))
    if rule == "pre_gap":
        return pd.notna(day.pre_signal_close) and pd.notna(day.prev_close)
    return True


def ideas(day, rule):
    """按时间顺序给出候选订单，只允许使用入场前已收盘的数据。"""
    left, right = day.L, day.S
    rth = np.flatnonzero(left.session.eq("RTH"))
    first, last = rth[0], rth[-1]
    if rule in RTH_RULES:
        side, target = signals(rule, left.iloc[rth], right.iloc[rth])
        return [Entry(rth[i + 1], int(side[i]), last, target=target[i])
                for i in np.flatnonzero(side) if i + 1 < len(rth) - 1]
    if rule == "cash":
        return []
    if rule in ("hold_long", "hold_short"):
        return [Entry(first, 1 if rule == "hold_long" else -1, last, stop=1.0, take_pct=np.inf)]
    if rule in ("pre_gap", "at_four"):
        signal = day.pre_signal_close if rule == "pre_gap" else day.night_signal_close
        buy = first if rule == "pre_gap" else np.flatnonzero(left.session.eq("PRE"))[0]
        move = signal / day.prev_close - 1
        return ([Entry(buy, int(np.sign(move)), last, stop=.06, take_pct=np.inf, trail=.07)]
                if abs(move) >= .03 else [])

    night = np.flatnonzero(left.session.eq("OVN"))
    if rule == "night_vwap":
        if not day.bear:
            return []
        f = right.iloc[night]
        vwap = (((f.high + f.low + f.close) / 3 * f.volume).cumsum() / f.volume.cumsum()).to_numpy()
        close, high = f.close.to_numpy(), f.high.to_numpy()
        # 冻结上次夜盘实验的定义，每晚只观察第一次偏离。
        dips = np.flatnonzero(close[11:80] <= .985 * vwap[11:80])
        if not len(dips):
            return []
        i = 11 + dips[0]
        for j in range(i + 1, i + 4):
            if high[j] >= vwap[i]:
                break
            if close[j] > high[j-1] and close[j] < vwap[i]:
                return [Entry(night[j+1], -1, night[j+13], stop=.015, target=vwap[i])]
        return []
    move = left.close.iloc[night].to_numpy() / day.prev_close - 1
    valid = np.abs(move) >= .03
    if rule == "night_korea":
        known = day.korea_move.iloc[night].to_numpy()
        valid &= np.sign(move) * known >= .01
    # 20:55 开始的第 12 根收盘时已到 21:00 ET。
    ids = np.flatnonzero(valid & (np.arange(len(night)) >= 11))
    if not len(ids):
        return []
    i = ids[0]
    return [Entry(night[i]+1, int(np.sign(move[i])), last, stop=.06, take_pct=np.inf, trail=.07)]


@dataclass(frozen=True)
class Fill:
    day: date
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    qty: int
    entry_price: float
    exit_price: float
    pnl: float
    reason: str


def simulate(day, candidates, equity, multiplier, *, notional=NOTIONAL, max_trades=3):
    """先按开盘处理跳空，盘中同根触及两边按止损；移动止损从下一根开始生效。"""
    frames = {1: day.L, -1: day.S}
    quotes = {s: f[["open", "high", "low"]].to_numpy() for s, f in frames.items()}
    drag = np.where(day.L.session.eq("RTH"), .0005, .0015) * multiplier
    fee = FEE * multiplier
    trades, pnl, previous_exit = [], 0.0, -1
    for e in candidates:
        if len(trades) >= max_trades or pnl <= -equity * .015:
            break
        if e.bar <= previous_exit or e.bar >= e.last:
            continue
        bars = quotes[e.side]
        raw = bars[e.bar, 0]
        entry = raw * (1 + drag[e.bar])
        if entry > e.max_buy_price:
            continue
        qty = int(min(notional, equity + pnl - 2 * fee) / entry)
        if qty < 1:
            continue
        # 已知目标下使用开盘限价，预留买卖两侧成交成本和费用。
        if np.isfinite(e.target) and qty * (e.target * (1 - drag[e.bar]) - entry) <= 2 * fee:
            continue
        target = e.target if np.isfinite(e.target) else raw * (1 + e.take_pct)
        stop = raw * (1 - e.stop)
        k, sell, reason = e.last, bars[e.last, 0], "到时平仓"
        for j in range(e.bar, e.last):
            op, hi, lo = bars[j]
            if op >= target:
                k, sell, reason = j, target, "止盈"
                break
            if lo <= stop:
                k, sell, reason = j, min(op, stop), "止损"
                break
            if hi >= target:
                k, sell, reason = j, target, "止盈"
                break
            if e.trail:
                stop = max(stop, hi * (1 - e.trail))
        exit_price = sell * (1 - drag[k])
        net = qty * (exit_price - entry) - 2 * fee
        trades.append(Fill(day.date, "SOXL" if e.side == 1 else "SOXS", day.L.index[e.bar],
                           day.L.index[k], qty, entry, exit_price, net, reason))
        pnl += net
        previous_exit = k
    return trades


def replay(days, rule, multiplier=1):
    equity = CAPITAL
    daily, all_trades = [], []
    for d in days:
        trades = simulate(d, ideas(d, rule), equity, multiplier)
        daily.append(sum(t.pnl for t in trades))
        equity += daily[-1]
        all_trades.extend(trades)
    return daily, all_trades


def stats(result):
    daily, trades = result
    eq = CAPITAL + np.cumsum([0, *daily])
    return {"net": sum(daily), "dd": min(eq / np.maximum.accumulate(eq) - 1),
            "n": len(trades), "wins": sum(t.pnl > 0 for t in trades)}


def timing_checks(days, best, cut):
    """保留入场信号的日期和时段，改变成交延迟或对照方向。"""
    if best == "cash":
        return
    scenarios = (("晚一根成交", 1, 0, 1), ("晚一根成交且成本翻倍", 1, 0, 2),
                 ("同期固定买SOXL", 0, 1, 1), ("同期固定买SOXS", 0, -1, 1))
    for label, delay, side, multiplier in scenarios:
        # 回归策略的目标取自买入标的自身；反向对照不应直接照搬该价格。
        if side and best in ("night_vwap", "vwap_reclaim"):
            continue
        equity, daily, count = CAPITAL, [], 0
        for d in days:
            candidates = [replace(e, bar=e.bar + delay, side=side or e.side) for e in ideas(d, best)
                          if e.bar + delay < e.last]
            trades = simulate(d, candidates, equity, multiplier)
            net = sum(t.pnl for t in trades)
            equity += net
            daily.append(net)
            count += len(trades)
        print(f"{label} 共 {count} 笔，全期净利 ${sum(daily):.2f}，后段 ${sum(daily[cut:]):.2f}")


def compare(days, trade_count=0):
    hist = [d for d in days if d.date <= HISTORY_END and all(eligible(d, r) for r in RULES)]
    fresh = [d for d in days if d.date > HISTORY_END]
    if len(hist) < 2:
        raise ValueError("共同历史样本不足，请检查两只 ETF 的夜盘、盘前、常规时段及参考行情缓存")
    cut = max(1, int(.7 * len(hist)))
    early, late = hist[:cut], hist[cut:]
    print(f"共同完整交易日 {len(hist)} 天，{hist[0].date} 至 {hist[-1].date}。"
          f"选择期 {len(early)} 天，后段 {len(late)} 天从 {late[0].date} 起。")
    best, best_score = "cash", 0.0
    rows = []
    for rule in (*RULES, *CONTROLS):
        base = replay(hist, rule)
        stress = replay(hist, rule, 2)
        score = sum(stress[0][:cut])
        count = sum(t.day < late[0].date for t in stress[1])
        if rule in RULES and count >= 10 and score > best_score:
            best, best_score = rule, score
        recent = [d for d in fresh if eligible(d, rule)]
        new_net = None if not recent else stats(replay(recent, rule))["net"]
        stat = stats(base)
        rows.append({"规则": LABELS[rule], "前段2倍成本$": score, "后段净利$": sum(base[0][cut:]),
                     "后段2倍成本$": sum(stress[0][cut:]), "全期净利$": stat["net"],
                     "全期笔数": stat["n"], "日末回撤": f"{stat['dd']:.1%}",
                     "新增天数": len(recent), "新增净利$": new_net})
    print(pd.DataFrame(rows).to_string(index=False, na_rep="—", float_format=lambda v: f"{v:.2f}"))
    print(f"依选择期规则选出 {LABELS[best]} ({best})。")
    full = replay(hist, best)
    for label, result in [("全历史",full), ("后段", (full[0][cut:], [t for t in full[1] if t.day >= late[0].date]))]:
        stat = stats(result)
        split = {symbol: round(float(sum(t.pnl for t in result[1] if t.symbol == symbol)), 2)
                 for symbol in ("SOXL", "SOXS")}
        print(f"{label}净利 ${stat['net']:.2f}，{stat['wins']}/{stat['n']} 笔盈利，"
              f"最佳日 ${max(result[0]):.2f}，最差日 ${min(result[0]):.2f}，分标的净利 {split}")
    timing_checks(hist, best, cut)
    common_dates = {sample.date for sample in hist}
    extra = [d for d in days if d.date <= HISTORY_END and eligible(d, best) and d.date not in common_dates]
    if extra:
        s = stats(replay(extra, best))
        print(f"其余可单独回放所选规则的历史日期共 {len(extra)} 天，{s['n']} 笔，净利 ${s['net']:.2f}。")
    if trade_count > 0:
        for t in full[1][-trade_count:]:
            print(f"{t.symbol} {t.qty} 股 {t.entry_time} @ {t.entry_price:.4f} → "
                  f"{t.exit_time} @ {t.exit_price:.4f} {t.reason} 净利 ${t.pnl:.2f}")
    return best
