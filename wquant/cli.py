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


def _apply_mode(args, s):
    if getattr(args, "mode", None):
        s.raw["mode"] = args.mode


def cmd_scan(args, s):
    from .engine import scan
    _apply_mode(args, s)
    return 0 if scan(s, source=args.source) else 1


def cmd_execute(args, s):
    from .engine import execute
    _apply_mode(args, s)
    review = args.review or {"claude_code": "file", "api": "api", "off": "off"}[s.section("ai").get("mode", "claude_code")]
    execute(s, review=review, confirm_live=args.confirm_live, source=args.source)
    return 0


def cmd_run(args, s):
    """scan + execute。ai.mode=claude_code 时只做 scan，等 Claude Code 审核后再 execute。"""
    from .engine import execute, scan
    _apply_mode(args, s)
    ai_mode = "off" if args.no_ai else s.section("ai").get("mode", "claude_code")
    if not scan(s, source=args.source):
        return 1
    if ai_mode == "claude_code":
        print("\n下一步：在 Claude Code 里运行 /trade-review 审核候选，然后执行 `python -m wquant execute`")
        return 0
    execute(s, review="api" if ai_mode == "api" else "off", confirm_live=args.confirm_live, source=args.source)
    return 0


def cmd_status(args, s):
    from .engine import day_dir
    d = day_dir()
    c, dec = d / "candidates.json", d / "decisions.json"
    print(f"今天: {d.name}")
    if not c.exists():
        print("  还没有 scan")
        return 0
    packet = json.loads(c.read_text(encoding="utf-8"))
    print(f"  候选 {len(packet['candidates'])} 个（生成于 {packet['generated_at']}，环境 {packet['environment']}）")
    print(f"  审核结论: {'已写入' if dec.exists() else '还没有'}")
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

    def common(sp, live=False):
        sp.add_argument("--mode", choices=["dry_run", "paper", "webull"], help="覆盖 config.yaml 中的 mode")
        sp.add_argument("--source", choices=["webull", "cache", "synthetic"], default="webull")
        if live:
            sp.add_argument("--confirm-live", action="store_true", help="实盘下单的最后一道确认")

    common(sub.add_parser("scan", help="阶段1：生成今日候选包 decisions/<日期>/candidates.json"))
    e = sub.add_parser("execute", help="阶段2：读审核结论 → 风控 → 下单")
    common(e, live=True)
    e.add_argument("--review", choices=["file", "api", "off"], help="审核来源，默认按 ai.mode")
    r = sub.add_parser("run", help="scan + execute（claude_code 模式下只做 scan）")
    common(r, live=True)
    r.add_argument("--no-ai", action="store_true", help="跳过审核，纯规则执行")
    sub.add_parser("status", help="查看今天的候选和审核状态")

    args = p.parse_args(argv)
    s = load_settings(args.config)
    cmds = {"check": cmd_check, "fetch": cmd_fetch, "backtest": cmd_backtest, "scan": cmd_scan,
            "execute": cmd_execute, "run": cmd_run, "status": cmd_status}
    return cmds[args.cmd](args, s)


if __name__ == "__main__":
    sys.exit(main())
