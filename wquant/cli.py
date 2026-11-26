"""命令行入口：python -m wquant <command>"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal

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
    from datetime import datetime
    from .engine import day_dir, execute
    _apply_mode(args, s)
    review = args.review or "off"
    # 真实下单每天只执行一次（定时任务和手动运行撞在一起时防止重复下单）
    marker = day_dir() / f"executed_{s.mode}_{s.environment}.json"
    if s.mode == "webull" and marker.exists() and not args.force:
        print(f"今天已经执行过（{marker.name}），不再重复下单。确实要再执行一次请加 --force")
        return 0
    res = execute(s, review=review, confirm_live=args.confirm_live, source=args.source)
    if s.mode == "webull":
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"at": datetime.now().isoformat(timespec="seconds"),
                                      "orders": [{k: r.get(k) for k in ("symbol", "signal", "outcome")} for r in res]},
                                     ensure_ascii=False, indent=1))
    return 0


def cmd_run(args, s):
    """Run the semiconductor core and tactical layers using the same entry point as swing tactical."""
    args.what = "tactical"
    return cmd_swing(args, s)


def cmd_futu_check(args, s):
    from .futu_gateway import check_accounts

    return check_accounts()


def cmd_wechat(args, s):
    from getpass import getpass

    from . import clawbot, wechat

    if args.action == "watch":
        wechat.watch(args.config)
        return 0
    if args.action == "chat-login":
        try:
            clawbot.pairing()
        except (OSError, ValueError, KeyError, clawbot.ClawError) as exc:
            error = str(exc) if isinstance(exc, clawbot.ClawError) else type(exc).__name__
            print(f"微信绑定未完成（{error}）")
            return 1
        return 0
    if args.action == "chat-status":
        from . import gemini_chat
        from .codex_chat import binary

        account, sync = clawbot.state()
        print("微信 ClawBot 已绑定" if account else "微信 ClawBot 等待扫码绑定")
        print("收到过私聊" if sync.get("context") else "尚未收到私聊")
        if sync.get("last_poll_at"):
            print("微信收件接口已响应")
        if sync.get("last_reply_at"):
            print("微信接口已接受私聊回复")
        print(f"当前默认问答引擎：{clawbot.chat_provider()}")
        print(f"已配置 Gemini API Key（模型 {gemini_chat.model_name()}）" if clawbot.gemini_key() else "尚未配置 Gemini API Key")
        print("可调用本机官方 Gemini CLI（个人账号）" if gemini_chat.binary() else "未安装本机官方 Gemini CLI")
        print("可调用本机 Codex CLI" if binary() else "未找到本机 Codex CLI")
        if clawbot.openai_key():
            print("已配置备用 OpenAI API Key")
        return 0
    if args.action == "chat-gemini-setup":
        import requests
        from . import gemini_chat

        if not sys.stdin.isatty():
            print("请在本机终端运行此命令，输入会隐藏。")
            return 1
        key = getpass("粘贴 Gemini API Key（来自 Google AI Studio）：").strip()
        if not key:
            print("未输入 Gemini API Key。")
            return 1
        try:
            gemini_chat.verify_api_key(key)
        except requests.RequestException as exc:
            print(f"Gemini API 验证失败（{type(exc).__name__}）。")
            return 1
        clawbot.save_gemini_key(key)
        clawbot.save_chat_provider("gemini")
        print(f"Gemini API Key 已保存并设为默认问答引擎（模型 {gemini_chat.model_name()}），可在微信私聊直接发问题验证。")
        return 0
    if args.action == "chat-gpt-setup":
        import requests

        if not sys.stdin.isatty():
            print("请在本机终端运行此命令，输入会隐藏。")
            return 1
        key = getpass("粘贴 OpenAI API Key：").strip()
        if not key:
            print("未输入 OpenAI API Key。")
            return 1
        try:
            response = requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"},
                                    timeout=10, allow_redirects=False)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"OpenAI API 验证失败（{type(exc).__name__}）。")
            return 1
        clawbot.save_openai_key(key)
        print("OpenAI API Key 已保存，可在 ClawBot 私聊发普通问题验证。")
        return 0
    if args.action == "status":
        queued, open_orders = wechat.counts()
        provider, token = wechat.read_channel()
        print(f"微信推送凭证已配置（{wechat.PROVIDER_NAMES[provider]}）" if token else "微信尚未绑定")
        print(f"待提交推送 {queued} 条，待跟踪 Webull 订单 {open_orders} 笔")
        return 0
    if args.action == "setup":
        if not sys.stdin.isatty():
            print("请在本机终端运行绑定命令，以隐藏输入的 Token。")
            return 1
        provider = args.provider or "pushplus"
        token_label = "PushPlus 消息 Token" if provider == "pushplus" else "WPush API Key"
        token = getpass(f"粘贴 {token_label}（输入不会显示）：").strip()
    else:
        provider, token = wechat.read_channel()
    if not token:
        print("缺少微信推送凭证，请先运行 wechat setup。")
        return 1
    try:
        receipt = wechat.push("SOXL/SOXS 微信通知测试", "这是通道测试，没有发送证券订单。", token,
                              provider=provider)
    except (OSError, ValueError, wechat.PushError) as exc:
        error = str(exc) if isinstance(exc, wechat.PushError) else type(exc).__name__
        print(f"提交测试通知失败（{error}）")
        return 1
    if args.action == "setup":
        wechat.save_channel(provider, token)
    print(f"{wechat.PROVIDER_NAMES[provider]} 已接收测试消息，请在微信公众号确认收到。流水号 {receipt}")
    return 0


def cmd_full_buy_plan(args, s):
    from .full_buy_plan import plan_futu, plan_webull
    from .webull_client import WebullClient

    limit = args.limit
    if not limit.is_finite() or limit <= 0 or limit != limit.quantize(Decimal("0.01")):
        raise SystemExit("限价必须为正数，最多两位小数")
    prod = load_settings(args.config, environment="prod")
    status = 0
    print(f"{args.symbol} 计划买入限价 USD {limit}（只读预检）")
    for name, plan in (
        ("Futu", lambda: plan_futu(args.symbol, limit, args.session)),
        ("Webull", lambda: plan_webull(WebullClient(prod), args.webull_account_suffix,
                                       args.symbol, limit, args.session)),
    ):
        try:
            result = plan()
            print(name, json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            print(f"{name} 预检失败：{exc}")
            status = 1
    return status


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


def _max_dd(values: list[float]) -> float:
    peak, dd = None, 0.0
    for v in values:
        peak = v if peak is None else max(peak, v)
        dd = min(dd, v / peak - 1)
    return dd


def _bench(sym: str, start: str) -> float | None:
    """基准：从 start 那天收盘到最新收盘的涨跌（用本地日线缓存）。"""
    from .data import CACHE, load_csv
    p = CACHE / f"{sym}.csv"
    if not p.exists():
        return None
    df = load_csv(p)
    before = df[df.index.date <= __import__("datetime").date.fromisoformat(start)]
    if before.empty:
        return None
    return float(df.close.iloc[-1] / before.close.iloc[-1] - 1)


def cmd_compare(args, s):
    """两套系统并排对比：个股选股系统（max_capital）和 SOXL/SOXS 战术层（tactical.capital），各自独立记账。"""
    from datetime import datetime
    from . import journal
    from . import swing_live as SL
    from .dashboard import _account
    env = s.environment
    rows = []
    # ---- 个股系统
    try:
        a = _account(s)
        hist = [json.loads(x) for x in (journal.STATE / f"equity_{env}.jsonl").read_text().splitlines()] \
            if (journal.STATE / f"equity_{env}.jsonl").exists() else []
        fills = json.loads((journal.STATE / f"fills_{env}.json").read_text()) if (journal.STATE / f"fills_{env}.json").exists() else []
        start = min((f["time"][:10] for f in fills), default=None)
        days = len({f["time"][:10] for f in fills})
        rows.append({"name": "个股选股系统", "capital": a["capital"], "equity": a["equity"], "realized": a["realized"],
                     "unrealized": a["unrealized"], "trades": len(fills), "days": days, "start": start,
                     "dd": _max_dd([h["equity"] for h in hist] + [a["equity"]]),
                     "holding": ", ".join(f"{p['symbol']} {p['quantity']:g}" for p in a["positions"]) or "空仓"})
    except Exception as e:
        rows.append({"name": "个股选股系统", "error": f"{type(e).__name__}: {e}"[:200]})
    # ---- 战术层
    cap = float(SL.swing_cfg(s)["tactical"]["capital"])
    eqp = SL._equity_path(env)
    eqd = json.loads(eqp.read_text()) if eqp.exists() else {"equity": cap, "history": []}
    orders = []
    for f in sorted(journal.LOGS.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("event") == "swing_order" and r.get("status") not in ("error", "rejected", None):
                orders.append(r)
    states = sorted(journal.STATE.glob(f"swing_tactical_{env}_*.json"))
    first = states[0].stem.rsplit("_", 1)[-1] if states else None
    traded_days = {r["ts"][:10] for r in orders}
    held = SL.Books(env).data.get("tactical", {})
    rows.append({"name": "SOXL/SOXS 战术层", "capital": cap, "equity": eqd["equity"], "realized": eqd["equity"] - cap,
                 "unrealized": 0.0, "trades": len(orders), "days": len(traded_days), "start": first,
                 "dd": _max_dd([cap] + [h["equity"] for h in eqd.get("history", [])]),
                 "holding": ", ".join(f"{k} {v:g}" for k, v in held.items()) or "空仓",
                 "watched": len(states)})
    print(f"环境 {env}    {datetime.now():%Y-%m-%d %H:%M}（两套系统各自独立本金，按收益率比较）\n")
    for r in rows:
        if "error" in r:
            print(f"{r['name']}: 读取失败 {r['error']}\n")
            continue
        ret = r["equity"] / r["capital"] - 1
        print(f"【{r['name']}】 起始 {r['start'] or '-'}")
        print(f"  本金 ${r['capital']:,.0f}  权益 ${r['equity']:,.2f}  收益率 {ret:+.2%}"
              f"（已实现 ${r['realized']:+,.2f} / 浮动 ${r['unrealized']:+,.2f}）  最大回撤 {r['dd']:+.2%}")
        extra = f"，已运行 {r['watched']} 个交易日" if r.get("watched") else ""
        print(f"  成交 {r['trades']} 笔，{r['days']} 天有交易{extra}；当前持仓：{r['holding']}")
        if r["start"]:
            b = [(sym, _bench(sym, r["start"])) for sym in ("QQQ", "SOXX")]
            print("  同期基准：" + "  ".join(f"{sym} {v:+.2%}" for sym, v in b if v is not None))
        print()
    snap = {"ts": datetime.now().isoformat(timespec="seconds"),
            **{r["name"]: round(r["equity"], 2) for r in rows if "equity" in r}}
    journal.STATE.mkdir(parents=True, exist_ok=True)
    with open(journal.STATE / f"compare_{env}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(snap, ensure_ascii=False) + "\n")
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


def cmd_scalp(args, s):
    from .scalp import ScalpParams
    try:
        params = ScalpParams(**s.section("scalp"))
    except (TypeError, ValueError) as exc:
        print(f"scalp 配置无效，{exc}", file=sys.stderr)
        return 2
    if (args.action == "paper" or args.fetch) and not s.has_webull_credentials:
        print("当前 Webull 环境缺少行情凭据，请检查项目的 .env 配置", file=sys.stderr)
        return 1
    if args.action == "paper":
        from .scalp_paper import run_paper
        return run_paper(s, params, watch=args.watch)
    from .intraday_data import download, load
    from .scalp_data import sessions
    from .scalp_report import print_report
    if args.fetch:
        from .webull_client import WebullClient
        try:
            client = WebullClient(s)
            for symbol in ("SOXL", "SOXS"):
                download(client, symbol, "M5")
        except Exception as exc:
            print(f"Webull 分钟线更新失败（{type(exc).__name__}），可用 scalp backtest 回放本地缓存", file=sys.stderr)
            return 1
    long_bars = load("SOXL", adjust=False)
    short_bars = load("SOXS", adjust=False)
    if long_bars.empty or short_bars.empty:
        print("缺少 SOXL / SOXS 分钟线，请运行 scalp backtest --fetch")
        return 1
    observed = set(long_bars.index[long_bars.session == "RTH"].date)
    observed.update(short_bars.index[short_bars.session == "RTH"].date)
    days = list(sessions(long_bars, short_bars))
    if not days:
        print("两只 ETF 的缓存中没有完整对齐的常规交易日")
        return 1
    print(f"SOXL / SOXS 常规时段 5 分钟线，{days[0][0]} 至 {days[-1][0]}。有行情的 {len(observed)} 日中纳入 {len(days)} 日。")
    print(f"初始本金 ${params.capital:,.2f}，每次最多 ${params.notional:,.2f}，完整点差 {params.spread_bps:g} bp，"
          f"单边滑点 {params.slippage_bps:g} bp，每次成交费用 ${params.fee_per_order:.2f}。")
    print_report(days, params, args.trades)
    return 0


def cmd_pair_study(args, s):
    from pathlib import Path
    from .intraday_data import DIR
    from .pair_data import study_days
    from .pair_study import compare
    forward = Path(args.forward_dir) if args.forward_dir else DIR / "pair_forward"
    if args.forward_dir and not forward.is_dir():
        print(f"找不到新增行情目录 {forward}", file=sys.stderr)
        return 1
    try:
        if args.bear_rally or args.trend_close:
            if args.news:
                raise ValueError("--news 需要与 --sizing 或 --entry-times 配合使用")
            if args.trend_close:
                from .trend_close import compare_close as run_study
            else:
                from .bear_rally import compare_rallies as run_study
            run_study(forward if forward.is_dir() else None, args.trades)
            return 0
        days = study_days(forward if forward.is_dir() else None)
        if args.sizing or args.entry_times:
            from .news_capture import DEFAULT_ARCHIVE
            news = args.news or (DEFAULT_ARCHIVE if DEFAULT_ARCHIVE.exists() else None)
            if args.entry_times:
                from .clock_study import compare_clocks
                compare_clocks(days, news, args.trades)
            else:
                from .pair_sizing import compare_sizing
                compare_sizing(days, news, args.trades)
        elif args.news:
            raise ValueError("--news 需要与 --sizing 或 --entry-times 配合使用")
        elif args.night_pullback:
            from .night_pullback import compare_pullbacks
            compare_pullbacks(days, args.trades)
        else:
            compare(days, args.trades)
    except (FileNotFoundError, ValueError) as exc:
        print(f"策略比较未完成，{exc}", file=sys.stderr)
        return 1
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
    from dataclasses import replace
    from datetime import datetime
    from . import swing_live as SL
    profile = getattr(args, "profile", None)
    if profile:
        s = replace(s, environment=f"{s.environment}_{profile}")
    cfg = SL.swing_cfg(s)
    mode = args.mode or "dry_run"
    now = lambda: datetime.now(SL.ET)
    books = SL.Books(s.environment)
    if args.what == "status":
        print(f"环境 {s.environment}  系统权益 ${SL.system_equity(s):,.2f}  账本 {books.data}")
        trend = cfg["tactical"].get("trend_switch") or {}
        if trend.get("enabled"):
            cst = SL._load_core_state(s.environment)
            peak = cst.peak if cst.peak_day == str(SL.trading_day(now())) else 0.0
            trail = peak * (1 - trend["trail_pct"])
            print(f"双层状态: 持核心={cst.holding} 均价={cst.entry:.2f} 日内最高={peak:.2f} "
                  f"移动止损线={trail:.2f} 止损冷却={cst.stopped}"
                  f"({cst.stopped_day or '-'}) 最近三票={cst.last_votes}")
        day = SL.trading_day(now())
        st = SL._load_state(s.environment, day)
        print(f"战术层 {day}: " + ("\n  ".join(st.log) if st.log else "还没开始"))
        return 0
    if args.what == "study":
        from .swing_study import main as study_main
        study_main(fetch=getattr(args, "fetch", False))
        return 0
    if args.what == "backtest":
        from .data import load_bars
        from .swing import backtest_weight, ensemble_weight, yearly
        B = load_bars(s, [cfg["core"]["index"], cfg["core"]["long"]], source="webull", count=1200)
        X, L = B[cfg["core"]["index"]], B[cfg["core"]["long"]]
        r = backtest_weight(X, L, ensemble_weight(X), allocation=cfg["core"]["allocation"], start="2022-06-01")
        print("核心层:", r.metrics, "按年", yearly(r.equity))
        return 0
    md = SL.make_market(s, now, source=getattr(args, "market_source", None))
    print(f"[swing] 行情源: {type(md).__name__}")
    if mode == "futu-shadow":
        from .futu_gateway import FutuShadowBroker

        broker = FutuShadowBroker(cfg["order"])
        if not profile:
            s = replace(s, environment=broker.book_environment)
            books = SL.Books(s.environment)
    elif mode == "futu-sim":
        from .futu_gateway import FutuSimBroker

        broker = FutuSimBroker(cfg["order"], label=profile or "futu-sim")
        if not profile:
            s = replace(s, environment=broker.book_environment)
            books = SL.Books(s.environment)
    else:
        broker = SL.make_broker(s, mode, args.confirm_live, cfg["order"], label=profile or "paper")
    if mode == "dry_run" and not profile:
        s = replace(s, environment=f"{s.environment}_dry")
        books = SL.Books(s.environment)
    if args.what == "core":
        if not cfg["core"].get("enabled", True):
            print("核心层在 config.yaml 里是关闭的（swing.core.enabled: false）")
            return 0
        SL.core_rebalance(s, md, broker, books)
    elif args.what == "tactical":
        run = SL.run_tactical_watch if getattr(args, "watch", False) else SL.run_tactical_day
        run(s, md, broker, books, now)
    return 0


def cmd_paper_arena(args, s):
    from datetime import datetime
    from . import paper_arena as PA
    from . import swing_live as SL

    if args.action == "status":
        return PA.print_arena_status(s)
    if args.companions_only:
        selected = ["plan_b", "plan_orig"]
    elif args.profiles:
        selected = [x.strip() for x in args.profiles.split(",") if x.strip()]
        invalid = [x for x in selected if x not in PA.PROFILE_MAP]
        if invalid:
            raise SystemExit(f"未知模拟盘方案：{invalid}，可选：{list(PA.PROFILE_MAP)}")
    else:
        selected = [p.key for p in PA.PROFILES]

    now = lambda: datetime.now(SL.ET)
    md = SL.make_market(s, now, source=getattr(args, "market_source", None))
    print(f"[paper-arena] 行情源: {type(md).__name__}")
    PA.run_arena_watch(
        s,
        md,
        selected,
        now,
        a_mode=args.a_mode,
        b_mode=args.b_mode,
        orig_mode=args.orig_mode,
        confirm_live=args.confirm_live,
        watch=args.watch,
        poll_seconds=int(SL.swing_cfg(s)["tactical"].get("poll_seconds", 60)),
    )
    return 0


def cmd_status(args, s):
    args.what = "status"
    args.mode = "dry_run"
    return cmd_swing(args, s)


def cmd_regime_bt(args, s) -> int:
    from .regime_bt import main as regime_main
    regime_main(fetch=args.fetch)
    return 0


def cmd_fair_value(args, s) -> int:
    from .fair_value import backtest_fair_value, compute_daily_fair_values, merge_cfg
    from .fair_value_data import (
        load_event_context,
        load_fundamentals,
        load_fv_bars,
        load_regime_context,
        load_webull_live_quotes,
    )
    from .fair_value_report import format_backtest_markdown, format_terminal_table, save_daily_report

    fv_cfg = merge_cfg(s.section("fair_value"))
    target_syms = [x.upper() for x in (args.symbols or s.universe)]
    fetch_syms = list(dict.fromkeys(list(s.universe) + ["SPY", "QQQ", "SOXX", "SOXL"] + target_syms))

    print(f"[fair-value] 加载行情（来源: {args.source}，共 {len(fetch_syms)} 只）...")
    bars = load_fv_bars(s, fetch_syms, source=args.source, refresh=not args.no_fetch)
    if not bars:
        print("✗ 没有可用日线行情数据")
        return 1

    bt_summary = None
    if args.backtest or args.backtest_only:
        print("[fair-value] 运行滚动样本外回测（Walk-Forward，无未来函数）...")
        bt_summary = backtest_fair_value(bars, s.groups, symbols=target_syms, cfg=fv_cfg)
        print()
        print(format_backtest_markdown(bt_summary))
        if args.backtest_only:
            return 0
        print()

    fundamentals = {}
    live_quotes = {}
    if args.source != "synthetic":
        refresh_fund = (not args.no_fund_fetch) and (args.source != "cache")
        fundamentals = load_fundamentals(fetch_syms, refresh=refresh_fund)
        if not args.no_live and args.source != "cache":
            print("[fair-value] 拉取 Webull 24 小时实时报价（含夜盘 OVN / 盘前 PRE）...")
            live_quotes = load_webull_live_quotes(s, fetch_syms)
            if live_quotes:
                sessions = sorted({v["session"] for v in live_quotes.values()})
                print(f"[fair-value] 已获取 {len(live_quotes)} 只 Webull 实时报价（时段: {'/'.join(sessions)}）")

    regime_ctx = load_regime_context(
        bars=bars,
        live_quotes=live_quotes,
        refresh_kospi=(not args.no_live) and (args.source not in ("cache", "synthetic")),
    )
    ev_ctx = load_event_context(
        fundamentals=fundamentals,
        earnings_window_days=int(fv_cfg["earnings_window_days"]),
    )
    results = compute_daily_fair_values(
        bars=bars,
        groups=s.groups,
        symbols=target_syms,
        fundamentals=fundamentals,
        event_context=ev_ctx,
        live_quotes=live_quotes,
        regime_context=regime_ctx,
        cfg=fv_cfg,
    )
    if not results:
        print("✗ 未能计算出有效估值结果（请检查行情数据长度）")
        return 1

    print()
    print(format_terminal_table(results, event_context=ev_ctx, regime_context=regime_ctx))
    if not args.no_save:
        json_path, md_path = save_daily_report(
            results,
            event_context=ev_ctx,
            regime_context=regime_ctx,
            source=f"{args.source}+webull_live" if live_quotes else args.source,
            backtest_summary=bt_summary,
        )
        print(f"\n✓ 每日估值报告已保存:\n  Markdown: {md_path}\n  JSON:     {json_path}")
    return 0


def cmd_fv_t(args, s) -> int:
    from .fair_value_data import load_fv_bars
    from .fv_t import FvTParams, run_fv_t_backtest, run_swing_t_backtest
    from .fv_t_data import fetch_m5_universe, fv_t_symbols
    from .fv_t_live import fv_t_cfg, params_from_cfg, print_fv_t_status, run_fv_t_once, watch_fv_t

    cfg = fv_t_cfg(s)
    syms, grp_map = fv_t_symbols(s, cfg["groups"])

    if args.action == "status":
        print_fv_t_status(s, mode=args.mode or "webull")
        return 0

    if args.action == "backtest":
        fetch_syms = list(dict.fromkeys(["SPY", "QQQ", "SOXX"] + syms))
        bars = load_fv_bars(s, fetch_syms, source=args.source, refresh=args.fetch)
        p = params_from_cfg(cfg)
        res = run_swing_t_backtest(bars, syms, grp_map, p)
        if res.get("error"):
            print(f"✗ 回测失败: {res['error']}")
            return 1
        print("=== 1~5日动态公允价波段做 T 回测（1x 成本: 12bp 双边 + $0.20）===")
        print(json.dumps({k: v for k, v in res.items() if k != "recent_trades"}, ensure_ascii=False, indent=2))
        if args.stress:
            p2 = params_from_cfg(cfg)
            p2.spread_bps *= 2.0
            p2.slippage_bps *= 2.0
            p2.fee_per_order *= 2.0
            res2 = run_swing_t_backtest(bars, syms, grp_map, p2)
            print("\n=== 双倍成本压力测试（24bp 双边 + $0.40）===")
            summary2 = {k: res2.get(k) for k in (
                "total_pnl", "total_return_pct", "first_half_return_pct", "second_half_return_pct",
                "last_40d_return_pct", "max_drawdown_pct", "sharpe", "trades", "win_rate_pct", "profit_factor",
            )}
            print(json.dumps(summary2, ensure_ascii=False, indent=2))
        if args.trades:
            print(f"\n最近 {args.trades} 笔做 T 记录:")
            for t in res.get("recent_trades", [])[-args.trades:]:
                print(
                    f"  {t['entry_date']}→{t['exit_date']} {t['symbol']:<5} {t['qty']:>4d}股 "
                    f"${t['entry_price']:.2f}→${t['exit_price']:.2f} PnL ${t['pnl']:+7.2f} ({t['ret_pct']:+.2f}%) [{t['reason']}]"
                )
        return 0

    if args.action == "m5-backtest":
        bars = fetch_m5_universe(s, syms, days=args.days, refresh=args.fetch)
        res = run_fv_t_backtest(bars, syms, grp_map, FvTParams())
        print("=== 5分钟日内动态公允价做 T 回测（对比验证）===")
        print(json.dumps({k: v for k, v in res.items() if k != "recent_trades"}, ensure_ascii=False, indent=2))
        return 0

    if args.action == "run":
        mode = args.mode or "dry_run"
        if args.watch:
            watch_fv_t(s, mode=mode, confirm_live=args.confirm_live, source=args.source, poll_minutes=args.poll)
            return 0
        run_fv_t_once(
            s,
            mode=mode,
            confirm_live=args.confirm_live,
            source=args.source,
            no_fetch=args.no_fetch,
            no_fund_fetch=args.no_fund_fetch,
            no_live=args.no_live,
        )
        return 0

    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="wquant", description="Webull SOXL/SOXS 三票制双层策略与回测")
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

    sc = sub.add_parser("scan", help="独立个股规则：生成候选包")
    common(sc)
    sc.add_argument("--force", action="store_true", help="重新生成当天候选包")
    e = sub.add_parser("execute", help="独立个股规则：风控后执行候选")
    common(e, live=True)
    e.add_argument("--review", choices=["file", "off"], help=argparse.SUPPRESS)
    e.add_argument("--force", action="store_true", help="今天已经执行过也再执行一次")
    r = sub.add_parser("run", help="运行 SOXL/SOXS 双层策略（默认只打印订单）")
    r.add_argument("--mode", choices=["dry_run", "paper", "webull", "futu-shadow", "futu-sim"], default="dry_run")
    r.add_argument("--profile", help="独立模拟盘账本后缀（如 plan_b / plan_orig），省略则使用默认主账本")
    r.add_argument("--market-source", choices=["futu", "webull", "auto"], help="覆盖 config.yaml 中的 swing.market_source")
    r.add_argument("--confirm-live", action="store_true", help="实盘下单的最后一道确认")
    r.add_argument("--watch", action="store_true", help="连续监控每个完整交易时段")
    st_p = sub.add_parser("status", help="查看双层策略的仓位和运行状态")
    st_p.add_argument("--profile", help="查看指定独立模拟盘账本状态")
    pa = sub.add_parser("paper-arena", help="三套 SOXL/SOXS 模拟盘并排运行与对比（A 利率双开关 / B 防收割 / 原版基准）")
    pa_sub = pa.add_subparsers(dest="action", required=True)
    pa_sub.add_parser("status", help="并排展示三套模拟盘的权益、回撤、持仓与最新日志")
    pa_run = pa_sub.add_parser("run", help="共享同一份实时行情同步运行模拟盘")
    pa_run.add_argument("--watch", action="store_true", help="跨交易日连续监控")
    pa_run.add_argument("--companions-only", action="store_true",
                        help="只跑 B 版与原版基准（当 Webull 沙盒已有独立进程跑 A 版时避免重复下单）")
    pa_run.add_argument("--profiles", help="自定义要运行的方案，逗号分隔：plan_a,plan_b,plan_orig")
    pa_run.add_argument("--market-source", choices=["futu", "webull", "auto"], help="覆盖行情源（默认读 config.yaml）")
    pa_run.add_argument("--a-mode", choices=["webull", "paper", "dry_run"], default="webull")
    pa_run.add_argument("--b-mode", choices=["futu-sim", "paper", "dry_run"], default="paper")
    pa_run.add_argument("--orig-mode", choices=["paper", "dry_run"], default="paper")
    pa_run.add_argument("--confirm-live", action="store_true")
    sub.add_parser("futu-check", help="只读检查本机富途 OpenD 美股账户")
    wx = sub.add_parser("wechat", help="绑定、测试或运行微信交易提醒")
    wx.add_argument("action", choices=["setup", "test", "status", "watch", "chat-login", "chat-status", "chat-gpt-setup", "chat-gemini-setup"])
    wx.add_argument("--provider", choices=["pushplus", "wpush"], help="绑定时选择推送服务，默认 PushPlus")
    fp = sub.add_parser("full-buy-plan", help="按时段只读核对富途和 Webull 的全仓买入数量")
    fp.add_argument("symbol", choices=["SOXL", "SOXS"])
    fp.add_argument("--limit", type=Decimal, required=True, help="实际拟提交的美元限价")
    fp.add_argument("--session", choices=["regular", "extended", "overnight"], required=True)
    fp.add_argument("--webull-account-suffix", required=True, help="Webull 生产账户的唯一尾号")
    o = sub.add_parser("orders", help="查看 Webull 账户里本系统下的订单")
    o.add_argument("--days", type=int, default=7)
    o.add_argument("--all", action="store_true", help="显示账户里所有订单（共享沙盒里包括别人的）")
    it = sub.add_parser("intraday-bt", help="日内做 T 回测（5 分钟线，常规时段）")
    it.add_argument("--fetch", action="store_true", help="先从 Webull 下载 / 更新最近 180 天分钟线")
    it.add_argument("--long", default="SOXL")
    it.add_argument("--short", default="SOXS")
    it.add_argument("--ref", default="QQQ")
    it.add_argument("--slippage", type=float, default=0.05, help="单边滑点（%）")
    scalp = sub.add_parser("scalp", help="SOXL / SOXS 常规时段 5 分钟波段交易")
    actions = scalp.add_subparsers(dest="action", required=True)
    sb = actions.add_parser("backtest", help="逐日回放、分段验证与双倍成本压力测试")
    sb.add_argument("--fetch", action="store_true", help="先从 Webull 更新两只 ETF 的分钟线")
    sb.add_argument("--trades", type=int, default=0, help="打印最近 N 笔完整交易")
    sp = actions.add_parser("paper", help="读取 Webull 分时行情，在本地模拟成交")
    sp.add_argument("--watch", action="store_true", help="持续轮询行情，Ctrl-C 停止")
    pair = sub.add_parser("pair-study", help="比较仅交易 SOXL/SOXS 的夜盘及日内策略")
    pair.add_argument("--forward-dir", help="新增验证行情目录，默认读取 data/intraday/pair_forward")
    pair.add_argument("--trades", type=int, default=0, help="展示所选策略最近 N 笔历史模拟成交")
    study_mode = pair.add_mutually_exclusive_group()
    study_mode.add_argument("--sizing", action="store_true", help="回放 04:00 条件全仓及消息过滤规则")
    study_mode.add_argument("--entry-times", action="store_true", help="比较固定进场时点并检验延迟成交与仓位")
    study_mode.add_argument("--night-pullback", action="store_true", help="按行业趋势比较夜盘 VWAP 附近及下方入场")
    study_mode.add_argument("--bear-rally", action="store_true", help="回测下跌趋势中夜盘拉高、SOXS 均价下方买入")
    study_mode.add_argument("--trend-close", action="store_true", help="顺趋势单边夜盘低吸，持有到常规盘尾盘退出")
    pair.add_argument("--news", help="带公布和首次可用时间的行业新闻归档，默认读取 news-record 的归档")
    from .news_capture import add_arguments, cmd_news_record
    add_arguments(sub)
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
    sw = sub.add_parser("swing", help="半导体波段系统：core 每日调仓 / tactical 当天夜盘战术 / status / backtest / study")
    sw.add_argument("what", choices=["core", "tactical", "status", "backtest", "study"])
    sw.add_argument("--fetch", action="store_true", help="study 用：先更新 Yahoo 日线")
    sw.add_argument("--mode", choices=["dry_run", "paper", "webull", "futu-shadow", "futu-sim"], help="默认 dry_run 只打印")
    sw.add_argument("--profile", help="独立模拟盘账本后缀")
    sw.add_argument("--confirm-live", action="store_true")
    rg = sub.add_parser("regime-bt", help="趋势研判回测：上涨只做 SOXL、下跌只做 SOXS（1996 起，含利率周期和 VIX）")
    rg.add_argument("--fetch", action="store_true", help="先从 Yahoo 更新 ^SOX、SOXL、SOXS、VIX、美债收益率日线")
    fv = sub.add_parser("fair-value", help="三层每日开盘前公允价值报告（板块联动 + 量价均衡 + 基本面锚）")
    fv.add_argument("symbols", nargs="*", help="指定标的，默认读取 config.yaml 全部 universe")
    fv.add_argument("--source", choices=["yahoo", "webull", "cache", "synthetic"], default="yahoo",
                    help="日线来源，默认 yahoo（免密钥、自动前复权）")
    fv.add_argument("--no-fetch", action="store_true", help="日线仅读本地缓存，不联网更新")
    fv.add_argument("--no-fund-fetch", action="store_true", help="基本面/盘前价仅读本地缓存，不联网更新")
    fv.add_argument("--no-live", action="store_true", help="不拉取 Webull 24h 夜盘实时价与 KOSPI 实时涨跌")
    fv.add_argument("--backtest", action="store_true", help="同时运行滚动样本外（Walk-Forward）回测验证")
    fv.add_argument("--backtest-only", action="store_true", help="仅运行历史样本外回测，不生成今日报告")
    fv.add_argument("--no-save", action="store_true", help="只打印终端表格，不写入 decisions/<日期>/fair_value.*")
    fvt = sub.add_parser("fv-t", help="动态公允价低买高卖做 T（独立模拟账本 + Webull UAT 沙盒下单）")
    fvt_sub = fvt.add_subparsers(dest="action", required=True)
    fvt_bt = fvt_sub.add_parser("backtest", help="1~5日公允价波段做 T 严格样本外回测（含双倍成本压力测试）")
    fvt_bt.add_argument("--source", choices=["yahoo", "webull", "cache", "synthetic"], default="yahoo")
    fvt_bt.add_argument("--fetch", action="store_true", help="先联网更新日线")
    fvt_bt.add_argument("--stress", action="store_true", help="附带双倍成本压力测试")
    fvt_bt.add_argument("--trades", type=int, default=0, help="打印最近 N 笔交易")
    fvt_m5 = fvt_sub.add_parser("m5-backtest", help="5分钟日内公允价做 T 对比回测（读 Webull M5 缓存）")
    fvt_m5.add_argument("--fetch", action="store_true", help="先从 Webull 更新 5 分钟 K 线")
    fvt_m5.add_argument("--days", type=int, default=90)
    fvt_run = fvt_sub.add_parser("run", help="执行公允价低买高卖做 T（默认 dry_run；--mode webull 才往 UAT 沙盒下单）")
    fvt_run.add_argument("--mode", choices=["dry_run", "webull"], default="dry_run")
    fvt_run.add_argument("--confirm-live", action="store_true", help="实盘下单最后确认（UAT 沙盒不需要）")
    fvt_run.add_argument("--source", choices=["yahoo", "webull", "cache", "synthetic"], default="yahoo")
    fvt_run.add_argument("--no-fetch", action="store_true")
    fvt_run.add_argument("--no-fund-fetch", action="store_true")
    fvt_run.add_argument("--no-live", action="store_true")
    fvt_run.add_argument("--watch", action="store_true", help="美股常规时段持续轮询（盘中回归公允价立即止盈轮动）")
    fvt_run.add_argument("--poll", type=int, default=None, help="轮询间隔（分钟），默认读 config.yaml")
    fvt_st = fvt_sub.add_parser("status", help="查看 fv-t 独立模拟账本持仓与已实现盈亏")
    fvt_st.add_argument("--mode", choices=["dry_run", "webull"], default="webull")
    sub.add_parser("compare", help="个股系统 vs SOXL/SOXS 战术层：收益率、回撤、交易次数并排对比")
    db = sub.add_parser("dashboard", help="启动本地网页仪表盘（只读）")
    db.add_argument("--port", type=int, default=8765)

    args = p.parse_args(argv)
    s = load_settings(args.config)
    cmds = {"check": cmd_check, "futu-check": cmd_futu_check, "full-buy-plan": cmd_full_buy_plan, "wechat": cmd_wechat,
            "fetch": cmd_fetch, "backtest": cmd_backtest, "scan": cmd_scan,
            "execute": cmd_execute, "run": cmd_run, "status": cmd_status, "paper-arena": cmd_paper_arena, "orders": cmd_orders, "dashboard": cmd_dashboard, "intraday-bt": cmd_intraday, "compare": cmd_compare, "overnight-bt": cmd_overnight, "swing": cmd_swing,
            "scalp": cmd_scalp, "pair-study": cmd_pair_study, "news-record": cmd_news_record,
            "regime-bt": cmd_regime_bt, "fair-value": cmd_fair_value, "fv-t": cmd_fv_t}
    return cmds[args.cmd](args, s)


if __name__ == "__main__":
    sys.exit(main())
