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
from .analyst import Decision, fallback_decision, price_summary
from .broker import DryRunBroker, PaperBroker, WebullBroker
from .config import ROOT, Settings
from .data import load_bars
from .indicators import add_indicators
from .risk import RiskManager
from .strategy import Signal, generate_signals

DECISIONS = ROOT / "decisions"


@dataclass
class Account:
    equity: float
    cash: float
    positions: dict[str, dict]   # {symbol: {quantity, cost_price}}


class DecisionFile(BaseModel):
    date: str
    reviewer: str = "claude-code"
    decisions: list[dict]


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
        return Account(client.equity(), client.cash(), client.positions()), client, None, client.open_order_symbols()
    paper = PaperBroker(order_cfg, settings.section("backtest").get("initial_cash", 100_000))
    return Account(paper.equity(prices), paper.cash(), paper.positions()), None, paper, set()


# =====================================================================
# 阶段 1：scan
# =====================================================================
def scan(settings: Settings, source: str = "webull") -> Path | None:
    strat, ai_cfg, risk_cfg = settings.section("strategy"), settings.section("ai"), settings.section("risk")
    if RiskManager.kill_switch_on():
        print("检测到 KILL_SWITCH 文件，本次不运行。删除该文件后恢复。")
        journal.record("halt", reason="kill_switch")
        return None

    print(f"[scan] 模式={settings.mode}  环境={settings.environment}  数据源={source}")
    bars = load_bars(settings, settings.universe, source=source, count=400)
    frames = {s: add_indicators(df, strat) for s, df in bars.items()}
    prices = {s: float(df["close"].iloc[-1]) for s, df in frames.items()}
    acct, client, _, busy = _open_account(settings, source, prices)
    sod = journal.start_of_day_equity(acct.equity)

    highs = journal.trail_highs()
    held = {s: max(highs.get(s, 0), p.get("cost_price", 0), prices[s])
            for s, p in acct.positions.items() if s in frames}
    journal.save_trail_highs(held)

    signals = sorted(generate_signals(frames, held, strat), key=lambda x: x.action != "SELL")
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
    """读取 Claude Code 写的 decisions.json，并逐条校验。"""
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
    return out


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
    if not candidates:
        print("今天没有候选，无需执行。")
        return []

    if review == "file":
        decisions = load_decisions()
    elif review == "api":
        decisions = decide_with_api(settings, candidates)
    else:
        decisions = {c["symbol"]: Decision(action="APPROVE", confidence=1.0, conviction=1.0,
                                           reasoning="AI 关闭，按规则执行", key_risks=[]) for c in candidates}

    prices = {c["symbol"]: c["price"] for c in candidates}
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
            rd = rm.size_buy(sig.symbol, sig.price, acct.equity, cash, n_pos, dec.conviction, sod)
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
            if intent.side == "BUY":
                n_pos, cash = n_pos + 1, cash - intent.notional
            else:
                n_pos, cash = n_pos - 1, cash + intent.notional
        entry.update(outcome=res.get("status"), order=intent.__dict__, broker=res)
        journal.record("order", **entry)
        results.append(entry)

    return results
