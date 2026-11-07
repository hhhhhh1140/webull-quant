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
    strat = s.section("strategy")
    if args.no_market_filter:
        strat["market_filter"] = {"enabled": False}
    universe = [x.upper() for x in (args.symbols or s.universe)]
    index_syms = [x.upper() for x in (strat.get("market_filter") or {}).get("symbols", [])]
    bars = load_bars(s, list(dict.fromkeys(universe + index_syms)), source=args.source, count=bt.get("bars", 1200))
    if not bars:
        print("没有数据")
        return 1
    res = run_backtest(bars, strat, s.section("risk"), bt, universe=universe, groups=s.groups,
                       instruments={x: s.instrument(x) for x in universe})
    bench = buy_and_hold({k: v for k, v in bars.items() if k in universe}, bt.get("initial_cash", 100_000))
    mf = strat.get("market_filter") or {}
    print(f"大盘过滤: {'开 ' + ','.join(index_syms) + ' / ' + mf.get('ma', '') if mf.get('enabled') else '关'}")
    print("策略:", json.dumps(res.metrics, ensure_ascii=False, indent=2))
    print(f"等权买入持有基准: 总收益 {bench.iloc[-1] / bench.iloc[0] - 1:.2%}   "
          f"最大回撤 {(bench / bench.cummax() - 1).min():.2%}")
    if args.trades:
        for t in res.trades[-args.trades:]:
            print(f"  {t.symbol:6s} {t.entry_date.date()} {t.entry_price:8.2f} → {t.exit_date.date()} "
                  f"{t.exit_price:8.2f}  {t.qty:9.3f} 股  PnL {t.pnl:9.2f}  {t.reason}")
    return 0


def _apply_mode(args, s):
    if getattr(args, "mode", None):
        s.raw["mode"] = args.mode


def cmd_scan(args, s):
    from .engine import scan
    _apply_mode(args, s)
    return 0 if scan(s, source=args.source, force=args.force) else 1


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


def _my_order_ids() -> set[str]:
    """从日志里找出本系统下过的订单 ID（共享账户里还有别人的订单）。"""
    from .journal import LOGS
    ids = set()
    for f in sorted(LOGS.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            coid = ((rec.get("broker") or {}).get("order") or {}).get("client_order_id")
            if coid and rec.get("outcome") == "submitted":
                ids.add(coid)
    return ids


def cmd_orders(args, s):
    from .journal import owned
    from .webull_client import WebullClient
    c = WebullClient(s)
    mine = _my_order_ids()
    rows = []
    for grp in c.open_orders() + c.order_history(args.days):
        for o in grp.get("orders", [grp]):
            if args.all or o.get("client_order_id") in mine:
                rows.append(o)
    seen, uniq = set(), []
    for o in rows:
        if o.get("order_id") not in seen:
            seen.add(o.get("order_id"))
            uniq.append(o)
    label = "账户全部订单" if args.all else "本系统下的订单"
    print(f"环境 {s.environment}  账户 {c.account_id}  {label}（近 {args.days} 天）: {len(uniq)} 笔")
    for o in sorted(uniq, key=lambda x: x.get("place_time", ""), reverse=True):
        t = (o.get("place_time_at") or "")[:19].replace("T", " ")
        px = o.get("limit_price") or "MKT"
        fill = f"成交 {o.get('filled_quantity')} @ {float(o.get('filled_price') or 0):.2f}"
        print(f"  {t} UTC  {o.get('side'):4s} {o.get('symbol'):6s} {o.get('total_quantity'):>5s} 股  "
              f"限价 {px:<12s} {o.get('support_trading_session', ''):7s} {o.get('status'):12s} {fill}")
    led = owned(s.environment)
    print(f"\n系统持仓账本: {led or '空'}")
    return 0


def cmd_dashboard(args, s):
    from .dashboard import serve
    serve(s, port=args.port)
    return 0


def cmd_intraday(args, s):
    """下载 / 更新分钟线并跑日内回测。"""
    import pandas as pd
    from .intraday_bt import STRATEGIES, Params, buy_hold_intraday, load_days, run, summarize
    from .intraday_data import download, load
    if args.fetch:
        from .webull_client import WebullClient
        c = WebullClient(s)
        for sym in (args.long, args.short, args.ref):
            df = download(c, sym, "M5")
            print(f"{sym}: {len(df)} 根 5 分钟线，{df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}")
    days = load_days(load(args.long), load(args.short), load(args.ref))
    if not days:
        print("没有分钟数据，先加 --fetch")
        return 1
    p = Params(slippage=args.slippage / 100)
    print(f"{len(days)} 个交易日（{days[0]['day']} → {days[-1]['day']}），滑点 {args.slippage}% 单边，每笔 ${p.notional:,.0f}，止损 {p.stop_pct:.0%}")
    rows = {}
    for name, fn in STRATEGIES.items():
        tr, daily = run(days, fn, p)
        rows[name] = summarize(tr, daily, p)
    for key, label in (("L", f"基准：每天开盘买 {args.long}"), ("S", f"基准：每天开盘买 {args.short}")):
        bh = buy_hold_intraday(days, key, p)
        rows[label] = {"总盈亏": round(bh.sum()), "收益率": round(bh.sum() / p.capital, 4), "笔数": len(bh)}
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).T.to_string())
    return 0


def cmd_overnight(args, s):
    """夜盘方向 → 顺势持有 的跨时段回测。"""
    from dataclasses import replace
    from .intraday24 import P24, build_days, run24, stats
    from .intraday_data import download, load
    syms = (args.long, args.short, "EWY")
    if args.fetch:
        from .webull_client import WebullClient
        c = WebullClient(s)
        for sym in syms:
            download(c, sym, "M5")
    days = build_days(*(load(x) for x in syms))
    if not days:
        print("没有分钟数据，先加 --fetch")
        return 1
    p = P24(entry=args.entry, exit=args.exit, threshold=args.threshold / 100, stop_pct=args.stop / 100,
            reentry=not args.no_reentry, korea_confirm=args.korea / 100 if args.korea is not None else None)
    tr, dl = run24(days, p)
    print(f"{len(days)} 个交易日（{days[0]['day']} → {days[-1]['day']}）")
    print(f"{args.entry} ET 判断，{args.long} 相对昨收 ≥±{args.threshold}% 顺势，止损 {args.stop}%，"
          f"{'可买回' if p.reentry else '不买回'}，韩国确认 {args.korea if args.korea is not None else '不看'}，{args.exit} 平仓")
    for k, v in stats(tr, dl, p).items():
        print(f"  {k}: {v}")
    return 0


def cmd_swing(args, s):
    from datetime import datetime
    from . import swing_live as SL
    cfg = SL.swing_cfg(s)
    mode = args.mode or "dry_run"
    now = lambda: datetime.now(SL.ET)
    books = SL.Books(s.environment)
    if args.what == "status":
        print(f"环境 {s.environment}  系统权益 ${SL.system_equity(s):,.2f}  账本 {books.data}")
        day = SL.trading_day(now())
        st = SL._load_state(s.environment, day)
        print(f"战术层 {day}: " + ("\n  ".join(st.log) if st.log else "还没开始"))
        return 0
    if args.what == "backtest":
        from .data import load_bars
        from .swing import backtest_weight, ensemble_weight, yearly
        B = load_bars(s, [cfg["core"]["index"], cfg["core"]["long"]], source="webull", count=1200)
        X, L = B[cfg["core"]["index"]], B[cfg["core"]["long"]]
        r = backtest_weight(X, L, ensemble_weight(X), allocation=cfg["core"]["allocation"], start="2022-06-01")
        print("核心层:", r.metrics, "按年", yearly(r.equity))
        return 0
    from .webull_client import WebullClient
    client = WebullClient(s)
    md = SL.WebullMarket(client, now)
    broker = SL.make_broker(s, mode, args.confirm_live, cfg["order"])
    dry = mode != "webull"
    if args.what == "core":
        if not cfg["core"].get("enabled", True):
            print("核心层在 config.yaml 里是关闭的（swing.core.enabled: false）")
            return 0
        SL.core_rebalance(s, md, broker, books, dry=dry)
    elif args.what == "tactical":
        SL.run_tactical_day(s, md, broker, books, now, dry=dry)
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
    b.add_argument("--no-market-filter", action="store_true", help="关闭大盘过滤（用于对比）")

    def common(sp, live=False):
        sp.add_argument("--mode", choices=["dry_run", "paper", "webull"], help="覆盖 config.yaml 中的 mode")
        sp.add_argument("--source", choices=["webull", "cache", "synthetic"], default="webull")
        if live:
            sp.add_argument("--confirm-live", action="store_true", help="实盘下单的最后一道确认")

    sc = sub.add_parser("scan", help="阶段1：生成今日候选包 decisions/<日期>/candidates.json")
    common(sc)
    sc.add_argument("--force", action="store_true", help="今天已审核过也重新扫描（会覆盖候选包和审核结论）")
    e = sub.add_parser("execute", help="阶段2：读审核结论 → 风控 → 下单")
    common(e, live=True)
    e.add_argument("--review", choices=["file", "api", "off"], help="审核来源，默认按 ai.mode")
    r = sub.add_parser("run", help="scan + execute（claude_code 模式下只做 scan）")
    common(r, live=True)
    r.add_argument("--no-ai", action="store_true", help="跳过审核，纯规则执行")
    sub.add_parser("status", help="查看今天的候选和审核状态")
    o = sub.add_parser("orders", help="查看 Webull 账户里本系统下的订单")
    o.add_argument("--days", type=int, default=7)
    o.add_argument("--all", action="store_true", help="显示账户里所有订单（共享沙盒里包括别人的）")
    it = sub.add_parser("intraday-bt", help="日内做 T 回测（5 分钟线，常规时段）")
    it.add_argument("--fetch", action="store_true", help="先从 Webull 下载 / 更新最近 180 天分钟线")
    it.add_argument("--long", default="SOXL")
    it.add_argument("--short", default="SOXS")
    it.add_argument("--ref", default="QQQ")
    it.add_argument("--slippage", type=float, default=0.05, help="单边滑点（%）")
    on = sub.add_parser("overnight-bt", help="夜盘方向 → 顺势持有 的跨时段回测")
    on.add_argument("--fetch", action="store_true")
    on.add_argument("--long", default="SOXL")
    on.add_argument("--short", default="SOXS")
    on.add_argument("--entry", default="04:00", help="美东时间，夜盘内或盘前")
    on.add_argument("--exit", default="16:00", choices=["09:30", "16:00"])
    on.add_argument("--threshold", type=float, default=3.0, help="相对昨收涨跌（%%）")
    on.add_argument("--stop", type=float, default=6.0, help="止损（%%）")
    on.add_argument("--no-reentry", action="store_true")
    on.add_argument("--korea", type=float, default=None, help="要求 EWY 同方向至少多少（%%）")
    sw = sub.add_parser("swing", help="半导体波段系统：core 每日调仓 / tactical 当天夜盘战术（会一直运行到收盘）/ status / backtest")
    sw.add_argument("what", choices=["core", "tactical", "status", "backtest"])
    sw.add_argument("--mode", choices=["dry_run", "webull"], help="默认 dry_run 只打印；webull 才下单")
    sw.add_argument("--confirm-live", action="store_true")
    db = sub.add_parser("dashboard", help="启动本地网页仪表盘（只读）")
    db.add_argument("--port", type=int, default=8765)

    args = p.parse_args(argv)
    s = load_settings(args.config)
    cmds = {"check": cmd_check, "fetch": cmd_fetch, "backtest": cmd_backtest, "scan": cmd_scan,
            "execute": cmd_execute, "run": cmd_run, "status": cmd_status, "orders": cmd_orders, "dashboard": cmd_dashboard, "intraday-bt": cmd_intraday, "overnight-bt": cmd_overnight, "swing": cmd_swing}
    return cmds[args.cmd](args, s)


if __name__ == "__main__":
    sys.exit(main())
