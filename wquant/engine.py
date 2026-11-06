"""每日流程，拆成两个阶段：

  scan     数据 → 规则信号 → 生成候选包 decisions/<日期>/candidates.json
  (审核)   Claude Code 读候选包，写 decisions/<日期>/decisions.json
           （或 ai.mode=api 时由程序调用 Claude API；ai.mode=off 时全部按规则执行）
  execute  读决策 → 风控 → 下单 → 日志

拆开之后，审核这一步可以由 Claude Code 在订阅额度内完成，程序本身不调用任何付费 API。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from . import journal
from .analyst import Decision, HedgeProposal, MarketView, fallback_decision, price_summary
from .broker import DryRunBroker, PaperBroker, WebullBroker
from .config import ROOT, Settings
from .data import load_bars
from .indicators import add_indicators
from .risk import RiskManager
from .strategy import Signal, generate_signals, hedge_exit_reason, hold_days_between, index_weak, market_regime

DECISIONS = ROOT / "decisions"


def account_key(settings: Settings) -> str:
    return settings.environment if settings.mode == "webull" else settings.mode


@dataclass
class Account:
    equity: float
    cash: float
    positions: dict[str, dict]   # {symbol: {quantity, cost_price}}


class DecisionFile(BaseModel):
    date: str
    reviewer: str = "claude-code"
    market_view: dict | None = None
    hedges: list[dict] = []
    decisions: list[dict] = []


def last_market_view(before: str | None = None) -> dict | None:
    """最近一次（今天之前）写过的市场研判。"""
    before = before or str(date.today())
    for d in sorted(DECISIONS.glob("*/decisions.json"), reverse=True):
        if d.parent.name >= before:
            continue
        try:
            mv = json.loads(d.read_text(encoding="utf-8")).get("market_view")
        except Exception:
            continue
        if mv:
            return {**mv, "date": d.parent.name}
    return None


def day_dir(day: str | None = None) -> Path:
    return DECISIONS / (day or str(date.today()))


def _open_account(settings: Settings, source: str, prices: dict[str, float]):
    """返回 (Account, webull_client 或 None, paper_broker 或 None, 有未成交订单的股票)。"""
    order_cfg = settings.section("order")
    use_webull = settings.mode == "webull" or (
        settings.mode == "dry_run" and settings.has_webull_credentials and source == "webull")
    if use_webull:
        from .webull_client import WebullClient
        client = WebullClient(settings)
        busy = client.open_order_symbols()
        acct = Account(client.equity(), client.cash(), client.positions())
        risk_cfg = settings.section("risk")
        if risk_cfg.get("manage_only_own_positions", True):
            acct.positions = _own_positions(settings.environment, acct.positions, busy)
        cap = risk_cfg.get("max_capital")
        if cap:
            own_mv = sum(prices.get(s, p.get("last_price") or p["cost_price"]) * p["quantity"]
                         for s, p in acct.positions.items())
            acct.equity = min(acct.equity, float(cap))
            acct.cash = max(min(acct.cash, float(cap) - own_mv), 0.0)
        return acct, client, None, busy
    paper = PaperBroker(order_cfg, settings.section("backtest").get("initial_cash", 100_000))
    return Account(paper.equity(prices), paper.cash(), paper.positions()), None, paper, set()


def _own_positions(env: str, actual: dict[str, dict], busy: set[str]) -> dict[str, dict]:
    """只保留账本里记录的、系统自己买入的那部分持仓。"""
    ledger = journal.owned(env)
    mine = {}
    for sym, qty in list(ledger.items()):
        if sym in actual:
            mine[sym] = {**actual[sym], "quantity": min(qty, actual[sym]["quantity"])}
        elif sym not in busy:
            ledger.pop(sym)          # 订单没成交或已被平掉，从账本里移除
    journal.save_owned(env, ledger)
    return mine


def _update_ledger(env: str, side: str, symbol: str, qty: int) -> None:
    ledger = journal.owned(env)
    ledger[symbol] = ledger.get(symbol, 0) + (qty if side == "BUY" else -qty)
    journal.save_owned(env, ledger)


# =====================================================================
# 阶段 1：scan
# =====================================================================
def scan(settings: Settings, source: str = "webull", force: bool = False) -> Path | None:
    strat, ai_cfg, risk_cfg = settings.section("strategy"), settings.section("ai"), settings.section("risk")
    if (day_dir() / "decisions.json").exists() and not force:
        print("今天的候选已经审核过了（decisions.json 已存在），不再覆盖。确实要重新扫描请加 --force")
        return None
    if RiskManager.kill_switch_on():
        print("检测到 KILL_SWITCH 文件，本次不运行。删除该文件后恢复。")
        journal.record("halt", reason="kill_switch")
        return None

    print(f"[scan] 模式={settings.mode}  环境={settings.environment}  数据源={source}")
    index_syms = (strat.get("market_filter") or {}).get("symbols", [])
    load = list(dict.fromkeys(settings.universe + [s.upper() for s in index_syms]))
    bars = load_bars(settings, load, source=source, count=400)
    all_frames = {s: add_indicators(df, strat) for s, df in bars.items()}
    regime = market_regime(all_frames, strat)
    frames = {s: f for s, f in all_frames.items() if s in settings.universe}
    prices = {s: float(df["close"].iloc[-1]) for s, df in frames.items()}
    acct, client, _, busy = _open_account(settings, source, prices)
    sod = journal.start_of_day_equity(acct.equity)

    highs = journal.trail_highs()
    held = {s: max(highs.get(s, 0), p.get("cost_price", 0), prices[s])
            for s, p in acct.positions.items() if s in frames}
    journal.save_trail_highs(held)

    key = account_key(settings)
    hedge_cfg = settings.section("hedge")
    open_hedges = {s: h for s, h in journal.hedges(key).items() if s in held}
    prev_view = last_market_view()
    hedge_sells = []
    for s, h in open_hedges.items():
        row = frames[s].iloc[-1]
        r = hedge_exit_reason(row, hedge_cfg, held[s], hold_days_between(h.get("entry_date"), frames[s].index[-1]),
                              (prev_view or {}).get("regime"))
        if r:
            hedge_sells.append(Signal(s, "SELL", r, float(row["close"]), float(row["atr"])))

    inverse = {s for s in frames if settings.instrument(s)["inverse"]}
    weak = index_weak(all_frames, strat)
    normal_held = {s: v for s, v in held.items() if s not in open_hedges}
    signals = sorted(hedge_sells + generate_signals(frames, normal_held, strat, inverse=inverse,
                                                    entry_dates=journal.entry_dates(key), weak_index=weak),
                     key=lambda x: x.action != "SELL")
    # 已经作为对冲持有的标的，不再作为普通买入候选
    signals = [s for s in signals if not (s.action == "BUY" and s.symbol in open_hedges)]
    from .sentiment import compute as compute_sentiment
    senti = compute_sentiment(settings, all_frames, client)
    blocked = []
    if regime["risk_off"]:
        # 反向 ETF（如 SOXS）不受大盘过滤限制
        stop = [s for s in signals if s.action == "BUY" and not settings.instrument(s.symbol)["inverse"]]
        blocked = [{"symbol": s.symbol, "price": round(s.price, 4), "rule_reason": s.reason} for s in stop]
        signals = [s for s in signals if s not in stop]
    candidates = []
    for sig in signals:
        news = []
        if client:
            try:
                news = client.stock_news(sig.symbol, ai_cfg.get("news_items", 8))
            except Exception as e:  # 新闻失败不影响流程
                print(f"  {sig.symbol} 新闻获取失败: {e}")
        candidates.append({
            "symbol": sig.symbol,
            "group": settings.groups.get(sig.symbol),
            "instrument": settings.instrument(sig.symbol),
            "action": sig.action,
            "rule_reason": sig.reason,
            "price": round(sig.price, 4),
            "atr": round(sig.atr, 4),
            "has_open_order": sig.symbol in busy,
            "price_summary": price_summary(frames[sig.symbol]),
            "position": acct.positions.get(sig.symbol),
            "news": news,
        })

    packet = {
        "date": str(date.today()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": settings.environment,
        "mode": settings.mode,
        "data_source": source,
        "account": {"equity": acct.equity, "cash": acct.cash, "start_of_day_equity": sod,
                    "positions": acct.positions},
        "market": regime,
        "sentiment": senti,
        "last_market_view": prev_view,
        "hedge_rules": {**hedge_cfg, "open_hedges": open_hedges},
        "hedge_quotes": {s: {**price_summary(frames[s]), "atr": round(float(frames[s]["atr"].iloc[-1]), 4),
                             "instrument": settings.instrument(s), "held": s in held}
                         for s in hedge_cfg.get("instruments", []) if s in frames},
        "blocked_buys": blocked,
        "risk_limits": risk_cfg,
        "ai_rules": {"min_confidence": ai_cfg.get("min_confidence", 0.6),
                     "can_veto_exits": ai_cfg.get("can_veto_exits", False)},
        "candidates": candidates,
    }
    d = day_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "candidates.json"
    path.write_text(json.dumps(packet, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    journal.record("scan", path=str(path), n=len(candidates))

    print(f"账户权益 {acct.equity:,.2f}  现金 {acct.cash:,.2f}  持仓 {list(acct.positions)}")
    if regime["enabled"]:
        desc = "  ".join(f"{i['symbol']} {i['close']:.2f} vs {i['ma_name']} {i['ma']:.2f} ({i['pct_vs_ma']:+.1%})"
                         for i in regime["indices"])
        print(f"大盘: {'弱势，停止开新仓' if regime['risk_off'] else '正常'}   {desc}")
    br = senti["breadth"]
    if br.get("above_sma50") is not None:
        chg = br.get("above_sma50_change_5d")
        print(f"广度: {br['above_sma50']:.0%} 站上 SMA50（5 日变化 {chg:+.0%}），{br['above_sma20']:.0%} 站上 SMA20"
              if chg is not None else f"广度: {br['above_sma50']:.0%} 站上 SMA50")
    if prev_view:
        print(f"上次市场研判（{prev_view['date']}）: {prev_view['regime']}  置信度 {prev_view.get('confidence', 0):.2f}")
    for s, h in open_hedges.items():
        print(f"  当前对冲: {s}（{h.get('entry_date')} 买入）")
    for b in blocked:
        print(f"  [BUY 已拦截] {b['symbol']} @ {b['price']:.2f} — 大盘过滤")
    if not candidates:
        print("今天没有候选。")
    for c in candidates:
        print(f"  [{c['action']}] {c['symbol']} @ {c['price']:.2f} — {c['rule_reason']}")
    print(f"候选包已写入 {path.relative_to(ROOT)}")
    return path


# =====================================================================
# 审核：三种来源
# =====================================================================
def load_decisions(day: str | None = None) -> dict[str, Decision]:
    return load_review(day)[0]


def load_review(day: str | None = None) -> tuple[dict[str, Decision], MarketView | None, list[HedgeProposal]]:
    """读取 Claude Code 写的 decisions.json，逐条校验。返回 (个股结论, 市场研判, 对冲提议)。"""
    path = day_dir(day) / "decisions.json"
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(ROOT)} 不存在。请先让 Claude Code 审核（/trade-review）")
    f = DecisionFile.model_validate_json(path.read_text(encoding="utf-8"))
    if f.date != (day or str(date.today())):
        raise ValueError(f"decisions.json 的日期 {f.date} 与今天不符，拒绝执行过期决策")
    out = {}
    for raw in f.decisions:
        sym = str(raw.get("symbol", "")).upper()
        try:
            out[sym] = Decision.model_validate({k: v for k, v in raw.items() if k != "symbol"})
        except ValidationError as e:
            print(f"  {sym} 的决策格式不合法，按保守处理: {e.errors()[0]['msg']}")
    view = None
    if f.market_view:
        try:
            view = MarketView.model_validate(f.market_view)
        except ValidationError as e:
            print(f"  市场研判格式不合法，忽略: {e.errors()[0]['msg']}")
    hedges = []
    for h in f.hedges:
        try:
            hedges.append(HedgeProposal.model_validate({**h, "symbol": str(h.get("symbol", "")).upper()}))
        except ValidationError as e:
            print(f"  对冲提议格式不合法，忽略: {e.errors()[0]['msg']}")
    return out, view, hedges


def decide_with_api(settings: Settings, candidates: list[dict]) -> dict[str, Decision]:
    from .analyst import ApiAnalyst  # 只有 api 模式才会导入 anthropic
    a = ApiAnalyst(settings.section("ai"))
    return {c["symbol"]: a.review(c) for c in candidates}


# =====================================================================
# 阶段 2：execute
# =====================================================================
def execute(settings: Settings, review: str, confirm_live: bool = False, source: str = "webull") -> list[dict]:
    """review: file（读 Claude Code 写的决策）| api | off"""
    ai_cfg, risk_cfg, order_cfg = settings.section("ai"), settings.section("risk"), settings.section("order")
    rm = RiskManager(risk_cfg, settings.universe)
    if rm.kill_switch_on():
        print("检测到 KILL_SWITCH 文件，本次不执行。")
        return []

    cpath = day_dir() / "candidates.json"
    if not cpath.exists():
        raise FileNotFoundError("今天还没有 scan。先运行 `python -m wquant scan`")
    packet = json.loads(cpath.read_text(encoding="utf-8"))
    candidates = packet["candidates"]
    if (packet.get("market") or {}).get("risk_off"):
        is_blocked = lambda c: c["action"] == "BUY" and not settings.instrument(c["symbol"])["inverse"]
        dropped = [c["symbol"] for c in candidates if is_blocked(c)]
        candidates = [c for c in candidates if not is_blocked(c)]
        if dropped:
            print(f"大盘过滤生效，不执行买入: {dropped}")
    view, hedges = None, []
    if review == "file":
        decisions, view, hedges = load_review()
    elif review == "api":
        decisions = decide_with_api(settings, candidates)
    else:
        decisions = {c["symbol"]: Decision(action="APPROVE", confidence=1.0, conviction=1.0,
                                           reasoning="AI 关闭，按规则执行", key_risks=[]) for c in candidates}
    if view:
        print(f"市场研判: {view.regime}  置信度 {view.confidence:.2f} — {view.summary}")
        journal.record("market_view", **view.model_dump())
        if view.regime == "risk_off":
            is_blocked = lambda c: c["action"] == "BUY" and not settings.instrument(c["symbol"])["inverse"]
            dropped = [c["symbol"] for c in candidates if is_blocked(c)]
            candidates = [c for c in candidates if not is_blocked(c)]
            if dropped:
                print(f"市场研判偏空，不执行做多买入: {dropped}")
    if not candidates and not hedges:
        print("今天没有候选，无需执行。")
        return []

    prices = {c["symbol"]: c["price"] for c in candidates}
    prices.update({s: q["close"] for s, q in (packet.get("hedge_quotes") or {}).items()})
    acct, client, paper, busy = _open_account(settings, packet.get("data_source", source), prices)
    if settings.mode == "webull":
        broker = WebullBroker(settings, client, order_cfg, confirm_live=confirm_live)
    elif settings.mode == "paper":
        broker = paper
    else:
        broker = DryRunBroker(order_cfg)

    sod = journal.start_of_day_equity(acct.equity)
    print(f"[execute] 模式={settings.mode}  环境={settings.environment}  审核={review}")
    n_pos, cash = len(acct.positions), acct.cash
    groups = settings.groups
    held_groups = [groups.get(s) for s in acct.positions]
    results = []

    for c in candidates:
        sig = Signal(c["symbol"], c["action"], c["rule_reason"], c["price"], c["atr"])
        print(f"\n[{sig.action}] {sig.symbol} @ {sig.price:.2f} — {sig.reason}")
        if sig.symbol in busy:
            print("  已有未成交订单，跳过")
            continue

        dec = decisions.get(sig.symbol) or fallback_decision(sig, "没有找到这只股票的审核结论")
        if sig.action == "SELL" and dec.action == "REJECT" and not ai_cfg.get("can_veto_exits", False):
            print(f"  审核想否决卖出（{dec.reasoning}），但配置不允许否决止损，照常卖出")
            dec = fallback_decision(sig, "止损优先")
        print(f"  审核: {dec.action}  置信度 {dec.confidence:.2f}  仓位系数 {dec.conviction:.2f}  — {dec.reasoning}")
        if dec.key_risks:
            print(f"  风险点: {'; '.join(dec.key_risks)}")

        entry = {"symbol": sig.symbol, "signal": sig.action, "rule_reason": sig.reason, "price": sig.price,
                 "review_source": review, "review": dec.model_dump()}
        if dec.action == "REJECT" or (sig.action == "BUY" and dec.confidence < ai_cfg.get("min_confidence", 0.6)):
            print("  → 不执行")
            entry["outcome"] = "rejected_by_review"
            journal.record("decision", **entry)
            results.append(entry)
            continue

        if sig.action == "BUY":
            g = groups.get(sig.symbol)
            rd = rm.size_buy(sig.symbol, sig.price, acct.equity, cash, n_pos, dec.conviction, sod,
                             leverage=settings.instrument(sig.symbol)["leverage"], group=g,
                             group_count=held_groups.count(g))
        else:
            held = acct.positions.get(sig.symbol, {}).get("quantity", 0)
            rd = rm.check_sell(sig.symbol, held, sig.price)
        for n in rd.notes:
            print(f"  风控: {n}")
        if not rd.approved:
            entry.update(outcome="rejected_by_risk", risk_notes=rd.notes)
            journal.record("decision", **entry)
            results.append(entry)
            continue

        intent = rd.intent
        intent.reason = sig.reason
        print(f"  → {intent.side} {intent.quantity} 股 ≈ ${intent.notional:,.0f}")
        try:
            res = broker.submit(intent)
        except Exception as e:
            res = {"status": "error", "error": str(e)}
            print(f"  下单失败: {e}")
        if res.get("status") not in ("error", "rejected"):
            if settings.mode == "webull":
                _update_ledger(settings.environment, intent.side, intent.symbol, intent.quantity)
            journal.set_entry_date(account_key(settings), intent.symbol,
                                   str(date.today()) if intent.side == "BUY" else None)
            if intent.side == "SELL":
                journal.set_hedge(account_key(settings), intent.symbol, None)
            if intent.side == "BUY":
                n_pos, cash = n_pos + 1, cash - intent.notional
                held_groups.append(groups.get(intent.symbol))
            else:
                n_pos, cash = n_pos - 1, cash + intent.notional
        entry.update(outcome=res.get("status"), order=intent.__dict__, broker=res)
        journal.record("order", **entry)
        results.append(entry)

    # ---------------- 对冲：只在市场研判偏空且置信度足够时，按严格限制开仓 ----------------
    results += _execute_hedges(settings, rm, broker, packet, view, hedges, acct, busy, sod,
                               n_pos, cash, {r["symbol"] for r in results if r.get("signal") == "BUY"
                                             and r.get("outcome") in ("submitted", "filled", "dry_run")})
    return results


def _execute_hedges(settings, rm, broker, packet, view, hedges, acct, busy, sod, n_pos, cash, bought) -> list[dict]:
    cfg = settings.section("hedge")
    out = []
    if not hedges:
        return out
    key = account_key(settings)
    quotes = packet.get("hedge_quotes") or {}
    for h in hedges:
        entry = {"symbol": h.symbol, "signal": "HEDGE", "review": h.model_dump(),
                 "market_view": view.model_dump() if view else None}
        why = None
        if not cfg.get("enabled", False):
            why = "对冲功能未开启"
        elif view is None or view.regime != "risk_off":
            why = "市场研判不是 risk_off"
        elif view.confidence < cfg.get("min_confidence", 0.7):
            why = f"研判置信度 {view.confidence:.2f} 低于 {cfg.get('min_confidence', 0.7)}"
        elif h.symbol not in cfg.get("instruments", []):
            why = f"{h.symbol} 不在对冲工具名单里"
        elif h.symbol not in quotes:
            why = "候选包里没有这个对冲工具的报价"
        elif h.symbol in acct.positions or h.symbol in bought:
            why = "已经持有"
        elif h.symbol in busy:
            why = "已有未成交订单"
        print(f"\n[HEDGE] {h.symbol} 仓位系数 {h.conviction:.2f} — {h.reasoning}")
        if why:
            print(f"  → 不执行：{why}")
            entry["outcome"] = "hedge_rejected"
            entry["reason"] = why
            journal.record("decision", **entry)
            out.append(entry)
            continue
        price = float(quotes[h.symbol]["close"])
        rd = rm.size_hedge(h.symbol, price, acct.equity, cash, n_pos, h.conviction, cfg.get("max_position_pct", 0.03),
                           leverage=settings.instrument(h.symbol)["leverage"], start_of_day_equity=sod)
        for n in rd.notes:
            print(f"  风控: {n}")
        if not rd.approved:
            entry.update(outcome="rejected_by_risk", risk_notes=rd.notes)
            journal.record("decision", **entry)
            out.append(entry)
            continue
        intent = rd.intent
        intent.reason = "市场研判对冲"
        print(f"  → BUY {intent.quantity} 股 ≈ ${intent.notional:,.0f}（对冲，最多持有 {cfg.get('max_hold_days', 10)} 天）")
        try:
            res = broker.submit(intent)
        except Exception as e:
            res = {"status": "error", "error": str(e)}
            print(f"  下单失败: {e}")
        if res.get("status") not in ("error", "rejected"):
            if settings.mode == "webull":
                _update_ledger(settings.environment, "BUY", h.symbol, intent.quantity)
            journal.set_entry_date(key, h.symbol, str(date.today()))
            journal.set_hedge(key, h.symbol, {"entry_date": str(date.today()),
                                               "max_hold_days": cfg.get("max_hold_days", 10)})
            n_pos, cash = n_pos + 1, cash - intent.notional
        entry.update(outcome=res.get("status"), order=intent.__dict__, broker=res)
        journal.record("order", **entry)
        out.append(entry)
    return out
