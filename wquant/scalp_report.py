"""5 分钟波段策略的逐日回放与时间顺序验证。"""
from __future__ import annotations

from dataclasses import replace

from .scalp import run_session


def replay(days, params):
    equity = params.capital
    results = []
    for day, long_bars, short_bars, close_time in days:
        trades = []
        if equity > 0:
            trades, state = run_session(
                long_bars, short_bars, replace(params, capital=equity), close_time
            )
            if state["qty"]:
                raise ValueError(f"{day} 缺少收盘平仓记录")
        results.append((day, equity, trades))
        equity += sum(t["pnl"] for t in trades)
    return results


def metrics(results):
    initial = results[0][1]
    trades = [t for _, _, completed in results for t in completed]
    equity = peak = initial
    drawdown = 0.0
    for t in trades:
        equity += t["pnl"]
        peak = max(peak, equity)
        if peak > 0:
            drawdown = min(drawdown, equity / peak - 1)
    costs = sum(t["costs"] for t in trades)
    net = equity - initial
    count = len(trades)
    return {
        "交易日": len(results), "往返": count, "日均往返": count / len(results),
        "胜率": sum(t["pnl"] > 0 for t in trades) / count if count else None,
        "毛利$": net + costs, "成本$": costs, "净利$": net,
        "每笔净利$": net / count if count else None,
        "净收益率": net / initial if initial > 0 else None, "已平仓回撤": drawdown,
    }


def print_report(days, params, show_trades=0):
    import pandas as pd

    base = replay(days, params)
    split = max(1, int(len(base) * 0.7))
    rows = {"全期": metrics(base)}
    if split < len(base):
        rows["前段"] = metrics(base[:split])
        rows["后段验证"] = metrics(base[split:])
        stress = replace(params, spread_bps=2 * params.spread_bps,
                         slippage_bps=2 * params.slippage_bps,
                         fee_per_order=2 * params.fee_per_order)
        rows["后段验证（双倍成本）"] = metrics(replay(days[split:], replace(stress, capital=base[split][1])))
        print(f"按日期切分，验证期为 {base[split][0]} 至 {base[-1][0]}。")
    frame = pd.DataFrame.from_dict(rows, orient="index")
    for col in ("胜率", "净收益率", "已平仓回撤"):
        frame[col] = frame[col].map(lambda n: "—" if pd.isna(n) else f"{n:.2%}")
    print(frame.to_string(float_format=lambda n: f"{n:.2f}", na_rep="—"))
    if show_trades > 0:
        trades = [t for _, _, completed in base for t in completed]
        for t in trades[-show_trades:]:
            print(f"{t['symbol']} {t['qty']} 股  {t['entry_t']} @ {t['entry']:.4f} → "
                  f"{t['exit_t']} @ {t['exit']:.4f}  {t['reason']}  净利 ${t['pnl']:.2f}")
