"""命令行入口：python -m wquant <command>"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_settings


def cmd_check(args, s):
    print(f"环境: {s.environment}   区域: {s.region_id}   模式: {s.mode}   实盘开关: {s.live_trading_enabled}")
    if not s.has_webull_credentials:
        print("✗ 未配置 Webull 密钥（.env）")
        return 1
    from .webull_client import WebullClient
    c = WebullClient(s, interactive_auth=True)
    print(f"✓ 账户 {c.account_id}")
    print(f"  权益 {c.equity():,.2f}   现金 {c.cash():,.2f}")
    for sym, p in c.positions().items():
        print(f"  {sym:6s} {p['quantity']:>8.0f} 股  成本 {p['cost_price']:.2f}  现价 {p['last_price']:.2f}")
    return 0


def cmd_fetch(args, s):
    from .data import load_bars
    bars = load_bars(s, args.symbols or s.universe, source="webull", count=args.bars)
    for sym, df in bars.items():
        print(f"{sym:6s} {len(df):5d} 根  {df.index[0].date()} → {df.index[-1].date()}")
    return 0


def cmd_backtest(args, s):
    from .backtest import buy_and_hold, run_backtest
    from .data import load_bars
    bt = s.section("backtest")
    if args.source == "synthetic":
        print("⚠ 使用随机生成的假数据，只用来验证程序能跑通，结果没有任何参考意义\n")
    bars = load_bars(s, args.symbols or s.universe, source=args.source, count=bt.get("bars", 1200))
    if not bars:
        print("没有数据")
        return 1
    res = run_backtest(bars, s.section("strategy"), s.section("risk"), bt)
    bench = buy_and_hold(bars, bt.get("initial_cash", 100_000))
    print("策略:", json.dumps(res.metrics, ensure_ascii=False, indent=2))
    print(f"等权买入持有基准: 总收益 {bench.iloc[-1] / bench.iloc[0] - 1:.2%}   "
          f"最大回撤 {(bench / bench.cummax() - 1).min():.2%}")
    if args.trades:
        for t in res.trades[-args.trades:]:
            print(f"  {t.symbol:6s} {t.entry_date.date()} {t.entry_price:8.2f} → {t.exit_date.date()} "
                  f"{t.exit_price:8.2f}  {t.qty:5d} 股  PnL {t.pnl:9.2f}  {t.reason}")
    return 0


def cmd_run(args, s):
    from .engine import run_once
    if args.mode:
        s.raw["mode"] = args.mode
    run_once(s, source=args.source, confirm_live=args.confirm_live, use_ai=False if args.no_ai else None)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="wquant", description="Webull 量化交易系统（AI 辅助决策）")
    p.add_argument("--config", help="配置文件路径，默认 config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="检查 Webull 连接、账户和持仓")

    f = sub.add_parser("fetch", help="从 Webull 拉取日线并缓存")
    f.add_argument("symbols", nargs="*")
    f.add_argument("--bars", type=int, default=1200)

    b = sub.add_parser("backtest", help="回测规则层")
    b.add_argument("symbols", nargs="*")
    b.add_argument("--source", choices=["webull", "cache", "synthetic"], default="cache")
    b.add_argument("--trades", type=int, default=0, help="打印最后 N 笔交易")

    r = sub.add_parser("run", help="执行一次每日流程（信号 → AI → 风控 → 下单）")
    r.add_argument("--mode", choices=["dry_run", "paper", "webull"], help="覆盖 config.yaml 中的 mode")
    r.add_argument("--source", choices=["webull", "cache", "synthetic"], default="webull")
    r.add_argument("--no-ai", action="store_true", help="跳过 AI 审核，纯规则执行")
    r.add_argument("--confirm-live", action="store_true", help="实盘下单的最后一道确认")

    args = p.parse_args(argv)
    s = load_settings(args.config)
    return {"check": cmd_check, "fetch": cmd_fetch, "backtest": cmd_backtest, "run": cmd_run}[args.cmd](args, s)


if __name__ == "__main__":
    sys.exit(main())
