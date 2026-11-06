"""每日运行流程：数据 → 规则信号 → AI 审核 → 风控 → 执行 → 日志。"""
from __future__ import annotations

from dataclasses import dataclass

from . import journal
from .analyst import Analyst, Decision, _fallback_decision
from .broker import DryRunBroker, PaperBroker, WebullBroker
from .config import Settings
from .data import load_bars
from .indicators import add_indicators
from .risk import RiskManager
from .strategy import generate_signals


@dataclass
class Account:
    equity: float
    cash: float
    positions: dict[str, dict]   # {symbol: {quantity, cost_price}}


def run_once(settings: Settings, source: str = "webull", confirm_live: bool = False, use_ai: bool | None = None) -> list[dict]:
    strat, risk_cfg, ai_cfg = settings.section("strategy"), settings.section("risk"), settings.section("ai")
    order_cfg = settings.section("order")
    use_ai = ai_cfg.get("enabled", True) if use_ai is None else use_ai
    rm = RiskManager(risk_cfg, settings.universe)

    if rm.kill_switch_on():
        print("检测到 KILL_SWITCH 文件，本次不运行。删除该文件后恢复。")
        journal.record("halt", reason="kill_switch")
        return []

    print(f"模式={settings.mode}  环境={settings.environment}  AI={'开' if use_ai else '关'}  数据源={source}")

    # ---- 1. 数据 ----
    bars = load_bars(settings, settings.universe, source=source, count=400)
    frames = {s: add_indicators(df, strat) for s, df in bars.items()}
    prices = {s: float(df["close"].iloc[-1]) for s, df in frames.items()}

    # ---- 2. 账户与执行器 ----
    client = None
    if settings.mode == "webull" or (settings.mode == "dry_run" and settings.has_webull_credentials and source == "webull"):
        from .webull_client import WebullClient
        client = WebullClient(settings)
        acct = Account(client.equity(), client.cash(), client.positions())
        busy = client.open_order_symbols()
    else:
        paper = PaperBroker(order_cfg, settings.section("backtest").get("initial_cash", 100_000))
        acct = Account(paper.equity(prices), paper.cash(), paper.positions())
        busy = set()

    if settings.mode == "webull":
        broker = WebullBroker(settings, client, order_cfg, confirm_live=confirm_live)
    elif settings.mode == "paper":
        broker = paper
    else:
        broker = DryRunBroker(order_cfg)

    sod = journal.start_of_day_equity(acct.equity)
    print(f"账户权益 {acct.equity:,.2f}  现金 {acct.cash:,.2f}  持仓 {list(acct.positions)}  当日起始 {sod:,.2f}")

    # ---- 3. 规则信号 ----
    highs = journal.trail_highs()
    held = {}
    for s, p in acct.positions.items():
        if s in frames:
            held[s] = max(highs.get(s, 0), p.get("cost_price", 0), prices[s])
    journal.save_trail_highs(held)

    signals = generate_signals({s: f for s, f in frames.items()}, held, strat)
    # 卖出优先，释放资金
    signals.sort(key=lambda x: x.action != "SELL")
    if not signals:
        print("今天没有信号。")
    analyst = Analyst(ai_cfg) if use_ai and signals else None
    n_pos = len(acct.positions)
    cash = acct.cash
    results = []

    for sig in signals:
        print(f"\n[{sig.action}] {sig.symbol} @ {sig.price:.2f} — {sig.reason}")
        if sig.symbol in busy:
            print("  已有未成交订单，跳过")
            continue

        # ---- 4. AI 审核 ----
        if analyst:
            news = []
            if client:
                try:
                    news = client.stock_news(sig.symbol, ai_cfg.get("news_items", 8))
                except Exception as e:  # 新闻失败不影响决策流程
                    print(f"  新闻获取失败: {e}")
            dec: Decision = analyst.review(sig, frames[sig.symbol], news, acct.positions.get(sig.symbol))
        else:
            dec = Decision(action="APPROVE", confidence=1.0, conviction=1.0, reasoning="AI 关闭，按规则执行", key_risks=[])

        if sig.action == "SELL" and dec.action == "REJECT" and not ai_cfg.get("can_veto_exits", False):
            print(f"  AI 想否决卖出（{dec.reasoning}），但配置不允许否决止损，照常卖出")
            dec = _fallback_decision(sig, "止损优先")
        print(f"  AI: {dec.action}  置信度 {dec.confidence:.2f}  仓位系数 {dec.conviction:.2f}  — {dec.reasoning}")
        if dec.key_risks:
            print(f"  风险点: {'; '.join(dec.key_risks)}")

        entry = {"symbol": sig.symbol, "signal": sig.action, "rule_reason": sig.reason, "price": sig.price,
                 "ai": dec.model_dump()}
        if dec.action == "REJECT" or (sig.action == "BUY" and dec.confidence < ai_cfg.get("min_confidence", 0.6)):
            print("  → 不执行")
            entry["outcome"] = "rejected_by_ai"
            journal.record("decision", **entry)
            results.append(entry)
            continue

        # ---- 5. 风控 ----
        if sig.action == "BUY":
            rd = rm.size_buy(sig.symbol, sig.price, acct.equity, cash, n_pos, dec.conviction, sod)
        else:
            rd = rm.check_sell(sig.symbol, acct.positions[sig.symbol]["quantity"], sig.price)
        for n in rd.notes:
            print(f"  风控: {n}")
        if not rd.approved:
            entry["outcome"] = "rejected_by_risk"
            entry["risk_notes"] = rd.notes
            journal.record("decision", **entry)
            results.append(entry)
            continue

        # ---- 6. 执行 ----
        intent = rd.intent
        intent.reason = sig.reason
        print(f"  → {intent.side} {intent.quantity} 股 ≈ ${intent.notional:,.0f}")
        try:
            res = broker.submit(intent)
        except Exception as e:
            res = {"status": "error", "error": str(e)}
            print(f"  下单失败: {e}")
        if res.get("status") not in ("error", "rejected"):
            if intent.side == "BUY":
                n_pos += 1
                cash -= intent.notional
            else:
                n_pos -= 1
                cash += intent.notional
        entry.update(outcome=res.get("status"), order=intent.__dict__, broker=res)
        journal.record("order", **entry)
        results.append(entry)

    return results
