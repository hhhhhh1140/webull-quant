"""公允价值反复做 T 的 Webull 模拟盘执行器（独立账本隔离，绝不碰 swing 或账户里其他持仓）。

架构与安全保障：
  - 独立账本 state/fv_t_book_<env>.json：只管理本策略自己买入的股票，与 SOXL/SOXS 战术层完全隔离
  - 实证有效的 1~5 日动态公允价低买高卖（Walk-Forward 500日夏普 1.65，双倍成本仍 1.55）：
      * 卖出优先：回归公允价上方 (Z >= exit_z) 止盈 / 跌破硬止损 (-stop_pct) / 持有超 max_hold_days / SOXX 三票制进入熊市 (<= -2)
      * 低买开仓：非熊市趋势 (SOXX > -2) 且 KOSPI 未暴跌，按 Z <= entry_z 从最被低估的标的起分批限价买入
  - 支持单次执行（开盘前 / 盘中手动或定时跑）与 --watch 常规时段轮询（每 15 分钟刷新一次动态公允价，
    盘中一旦回升到公允价上方立即止盈落袋并轮动低吸下一只）
  - 安全锁：默认 mode=dry_run；--mode webull 在 UAT 沙盒下真实限价单；prod 环境强制检查 live_trading_enabled 与 --confirm-live
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from . import journal
from .broker import DryRunBroker, WebullBroker
from .config import ROOT, Settings
from .fair_value import StockFairValue, compute_daily_fair_values, merge_cfg
from .fair_value_data import (
    load_event_context,
    load_fundamentals,
    load_fv_bars,
    load_regime_context,
    load_webull_live_quotes,
)
from .fv_t import SwingTParams
from .fv_t_data import fv_t_symbols, is_rth_open
from .risk import OrderIntent

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
STATE_DIR = ROOT / "state"


def fv_t_cfg(settings: Settings) -> dict[str, Any]:
    raw = settings.section("fv_t")
    defaults: dict[str, Any] = {
        "groups": ["semis", "memory", "optical"],
        "capital": 10_000.0,
        "notional_per_trade": 2_500.0,
        "max_positions": 4,
        "max_per_group": 2,
        "entry_z": -0.60,
        "exit_z": 0.10,
        "stop_pct": 0.06,
        "max_hold_days": 5,
        "require_soxx_above": -2,
        "kospi_crash_pct": -2.0,
        "daily_loss_pct": 0.03,
        "poll_minutes": 15,
        "order": {
            "type": "LIMIT",
            "limit_offset_bps": 20,
            "time_in_force": "DAY",
            "trading_session": "ALL_DAY",
        },
    }
    out = {**defaults, **{k: v for k, v in raw.items() if k != "order"}}
    out["order"] = {**defaults["order"], **(raw.get("order") or {})}
    return out


def params_from_cfg(cfg: dict[str, Any]) -> SwingTParams:
    return SwingTParams(
        capital=float(cfg["capital"]),
        notional_per_trade=float(cfg["notional_per_trade"]),
        max_positions=int(cfg["max_positions"]),
        max_per_group=int(cfg["max_per_group"]),
        entry_z=float(cfg["entry_z"]),
        exit_z=float(cfg["exit_z"]),
        stop_pct=float(cfg["stop_pct"]),
        max_hold_days=int(cfg["max_hold_days"]),
        require_soxx_above=int(cfg["require_soxx_above"]),
        kospi_crash_pct=float(cfg["kospi_crash_pct"]),
    )


class FvTBook:
    """独立账本：记录本做 T 策略自己的现金、持仓与已实现盈亏，绝不干扰 swing 仓位。"""

    def __init__(self, env: str, capital: float = 10_000.0, path: Path | None = None):
        self.env = env
        self.path = path or (STATE_DIR / f"fv_t_book_{env}.json")
        self.trades_path = self.path.parent / f"fv_t_trades_{env}.jsonl"
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "env": env,
                "initial_capital": float(capital),
                "cash": float(capital),
                "realized_pnl": 0.0,
                "positions": {},       # {sym: {qty, cost_price, group, entry_date, entry_fv, entry_z}}
                "stopped_today": {},   # {date_str: [sym, ...]}
                "daily_pnl": {},       # {date_str: float}
                "updated_at": None,
            }

    @property
    def cash(self) -> float:
        return float(self.data.get("cash", 0.0))

    @property
    def positions(self) -> dict[str, dict[str, Any]]:
        return self.data.setdefault("positions", {})

    def equity(self, prices: dict[str, float]) -> float:
        mv = sum(
            float(p["qty"]) * float(prices.get(sym, p["cost_price"]))
            for sym, p in self.positions.items()
        )
        return self.cash + mv

    def daily_pnl(self, today: str) -> float:
        return float((self.data.get("daily_pnl") or {}).get(today, 0.0))

    def stopped_today(self, today: str) -> set[str]:
        return set((self.data.get("stopped_today") or {}).get(today, []))

    def record_buy(self, sym: str, qty: int, price: float, group: str, entry_fv: float, entry_z: float, today: str) -> None:
        cost = qty * price
        self.data["cash"] = round(self.cash - cost, 2)
        self.positions[sym] = {
            "qty": int(qty),
            "cost_price": round(price, 4),
            "group": group,
            "entry_date": today,
            "entry_fv": round(entry_fv, 2),
            "entry_z": round(entry_z, 2),
        }
        self.save()

    def record_sell(self, sym: str, price: float, reason: str, today: str) -> dict[str, Any]:
        pos = self.positions.pop(sym)
        qty = int(pos["qty"])
        cost_px = float(pos["cost_price"])
        proceeds = qty * price
        pnl = round(qty * (price - cost_px), 2)
        self.data["cash"] = round(self.cash + proceeds, 2)
        self.data["realized_pnl"] = round(float(self.data.get("realized_pnl", 0.0)) + pnl, 2)
        dp = self.data.setdefault("daily_pnl", {})
        dp[today] = round(float(dp.get(today, 0.0)) + pnl, 2)
        if reason == "stop":
            st = self.data.setdefault("stopped_today", {})
            st.setdefault(today, [])
            if sym not in st[today]:
                st[today].append(sym)
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "symbol": sym,
            "group": pos["group"],
            "qty": qty,
            "entry_date": pos["entry_date"],
            "exit_date": today,
            "entry_price": cost_px,
            "exit_price": round(price, 4),
            "pnl": pnl,
            "ret_pct": round((price / cost_px - 1.0) * 100.0, 2) if cost_px > 0 else 0.0,
            "reason": reason,
        }
        self.trades_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trades_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.save()
        return rec

    def save(self) -> None:
        self.data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")


def _hold_bdays(entry_date_str: str, today: date) -> int:
    try:
        ed = date.fromisoformat(entry_date_str)
        return max(0, int(np.busday_count(ed, today)))
    except Exception:
        return 0


def plan_fv_t_actions(
    fv_items: list[StockFairValue],
    book: FvTBook,
    regime_ctx: dict[str, Any],
    event_ctx: dict[str, Any],
    cfg: dict[str, Any],
    today: date | None = None,
    open_order_symbols: set[str] | None = None,
) -> list[dict[str, Any]]:
    """纯函数决策器：根据当前实时公允价值、持仓账本、市场趋势与财报窗口生成买卖意图列表。"""
    ref_day = today or date.today()
    today_str = str(ref_day)
    busy = open_order_symbols or set()
    by_sym = {item.symbol: item for item in fv_items}

    soxx_votes = regime_ctx.get("soxx_votes")
    kospi_pct = regime_ctx.get("kospi_move_pct")
    earnings_flags = (event_ctx or {}).get("earnings_flags") or {}

    bear_block = (soxx_votes is not None) and (int(soxx_votes) <= int(cfg["require_soxx_above"]))
    kospi_block = (kospi_pct is not None) and (float(kospi_pct) <= float(cfg["kospi_crash_pct"]))
    loss_halted = book.daily_pnl(today_str) <= -float(cfg["capital"]) * float(cfg["daily_loss_pct"])

    actions: list[dict[str, Any]] = []
    # 1) 卖出优先（止损 / 熊市趋势撤退 / 回归公允价高卖 / 持仓超时）
    remaining_positions = dict(book.positions)
    projected_cash = book.cash
    for sym, pos in list(book.positions.items()):
        if sym in busy or sym not in by_sym:
            continue
        item = by_sym[sym]
        px = float(item.ref_price)
        cost_px = float(pos["cost_price"])
        ret_now = (px / cost_px - 1.0) if cost_px > 0 else 0.0
        held = _hold_bdays(str(pos.get("entry_date", today_str)), ref_day)

        reason = None
        if ret_now <= -float(cfg["stop_pct"]):
            reason = f"stop ({ret_now:+.1%})"
        elif bear_block:
            reason = f"bear_regime (SOXX={soxx_votes:+d})"
        elif item.z_score >= float(cfg["exit_z"]):
            reason = f"revert_fv (Z={item.z_score:+.2f}, FV=${item.fair_value:.2f})"
        elif held >= int(cfg["max_hold_days"]):
            reason = f"timeout ({held}d)"

        if reason:
            actions.append(
                {
                    "side": "SELL",
                    "symbol": sym,
                    "group": pos["group"],
                    "quantity": int(pos["qty"]),
                    "price": px,
                    "fair_value": item.fair_value,
                    "z_score": item.z_score,
                    "reason_code": reason.split()[0],
                    "reason": reason,
                }
            )
            remaining_positions.pop(sym, None)
            projected_cash += int(pos["qty"]) * px

    # 2) 低买扫描（按 Z 值最深排序）
    if bear_block or kospi_block or loss_halted:
        return actions

    stopped_set = book.stopped_today(today_str)
    max_pos = int(cfg["max_positions"])
    max_per_grp = int(cfg["max_per_group"])
    notional = float(cfg["notional_per_trade"])
    entry_z = float(cfg["entry_z"])

    candidates = [
        item for item in fv_items
        if item.symbol not in remaining_positions
        and item.symbol not in busy
        and item.symbol not in stopped_set
        and item.z_score <= entry_z
        and (earnings_flags.get(item.symbol) or {}).get("status") != "EARNINGS_SOON"
    ]
    candidates.sort(key=lambda x: x.z_score)

    for item in candidates:
        if len(remaining_positions) >= max_pos:
            break
        grp_cnt = sum(1 for p in remaining_positions.values() if p["group"] == item.group)
        if grp_cnt >= max_per_grp:
            continue
        px = float(item.ref_price)
        if px <= 0:
            continue
        qty = int(min(notional, projected_cash) // px)
        if qty <= 0:
            continue
        actions.append(
            {
                "side": "BUY",
                "symbol": item.symbol,
                "group": item.group,
                "quantity": qty,
                "price": px,
                "fair_value": item.fair_value,
                "z_score": item.z_score,
                "reason_code": "undervalued_fv",
                "reason": f"低吸做T (Z={item.z_score:+.2f}, 偏离{item.mispricing_pct:+.1f}%, FV=${item.fair_value:.2f})",
            }
        )
        remaining_positions[item.symbol] = {"group": item.group, "qty": qty, "cost_price": px}
        projected_cash -= qty * px

    return actions


def make_fv_t_broker(settings: Settings, mode: str, confirm_live: bool, order_cfg: dict[str, Any]):
    if mode != "webull":
        return DryRunBroker(order_cfg), None
    if settings.environment == "prod":
        if not settings.live_trading_enabled:
            raise RuntimeError("WEBULL_ENVIRONMENT=prod，但 config.yaml 的 live_trading_enabled 不是 true，拒绝实盘下单")
        if not confirm_live:
            raise RuntimeError("实盘下单需要在命令行加 --confirm-live")
    from .webull_client import WebullClient

    client = WebullClient(settings)
    broker = WebullBroker(settings, client, order_cfg, confirm_live=confirm_live)
    return broker, client


def run_fv_t_once(
    settings: Settings,
    mode: str = "dry_run",
    confirm_live: bool = False,
    source: str = "yahoo",
    no_fetch: bool = False,
    no_fund_fetch: bool = False,
    no_live: bool = False,
) -> dict[str, Any]:
    """执行一次完整的公允价值做 T 扫描与下单（dry_run 只打印；webull 在 UAT 沙盒下真实限价单）。"""
    cfg = fv_t_cfg(settings)
    syms, grp_map = fv_t_symbols(settings, cfg["groups"])
    fetch_syms = list(dict.fromkeys(list(settings.universe) + ["SPY", "QQQ", "SOXX", "SOXL"] + syms))

    bars = load_fv_bars(settings, fetch_syms, source=source, refresh=not no_fetch)
    if not bars:
        raise RuntimeError("没有可用行情数据")

    fundamentals = {}
    live_quotes = {}
    if source != "synthetic":
        fundamentals = load_fundamentals(fetch_syms, refresh=(not no_fund_fetch) and (source != "cache"))
        if not no_live and source != "cache":
            live_quotes = load_webull_live_quotes(settings, fetch_syms)

    regime_ctx = load_regime_context(
        bars=bars,
        live_quotes=live_quotes,
        refresh_kospi=(not no_live) and (source not in ("cache", "synthetic")),
    )
    event_ctx = load_event_context(fundamentals=fundamentals)
    fv_cfg = merge_cfg(settings.section("fair_value"))
    fv_items = compute_daily_fair_values(
        bars=bars,
        groups=settings.groups,
        symbols=syms,
        fundamentals=fundamentals,
        event_context=event_ctx,
        live_quotes=live_quotes,
        regime_context=regime_ctx,
        cfg=fv_cfg,
    )

    env_key = settings.environment if mode == "webull" else f"dry_{settings.environment}"
    book = FvTBook(env=env_key, capital=float(cfg["capital"]))
    broker, client = make_fv_t_broker(settings, mode=mode, confirm_live=confirm_live, order_cfg=cfg["order"])
    busy = client.open_order_symbols() if client is not None else set()

    today_et = datetime.now(ET).date()
    today_str = str(today_et)
    actions = plan_fv_t_actions(
        fv_items=fv_items,
        book=book,
        regime_ctx=regime_ctx,
        event_ctx=event_ctx,
        cfg=cfg,
        today=today_et,
        open_order_symbols=busy,
    )

    print(f"[fv-t] 模式={mode} 环境={settings.environment} 账本={book.path.name}")
    if regime_ctx.get("summary"):
        print(f"★ 市场状态: {regime_ctx['summary']}")
    prices = {it.symbol: it.ref_price for it in fv_items}
    print(
        f"  账本权益: ${book.equity(prices):,.2f} | 现金: ${book.cash:,.2f} | "
        f"已实现: ${book.data.get('realized_pnl', 0.0):+,.2f} | 持仓: {len(book.positions)}/{cfg['max_positions']}"
    )

    if not actions:
        reasons = []
        sv = regime_ctx.get("soxx_votes")
        kp = regime_ctx.get("kospi_move_pct")
        if sv is not None and int(sv) <= int(cfg["require_soxx_above"]):
            reasons.append(f"SOXX 三票={sv:+d} 处于下跌趋势，暂停低买")
        if kp is not None and float(kp) <= float(cfg["kospi_crash_pct"]):
            reasons.append(f"KOSPI {kp:+.2f}% 急跌冲击，暂停低买")
        if not reasons:
            reasons.append(f"暂无触发买入 (Z <= {cfg['entry_z']:+.2f}) 或卖出 (Z >= {cfg['exit_z']:+.2f}) 的标的")
        print(f"  本轮无交易动作：{'；'.join(reasons)}")
        return {"mode": mode, "regime": regime_ctx, "actions": [], "executed": []}

    executed: list[dict[str, Any]] = []
    for act in actions:
        intent = OrderIntent(
            symbol=act["symbol"],
            side=act["side"],
            quantity=float(act["quantity"]),
            price=float(act["price"]),
            reason=act["reason"],
        )
        print(f"  → {act['side']:4s} {act['symbol']:<5} {act['quantity']:>4d} 股 @ ${act['price']:.2f} | {act['reason']}")
        res = broker.submit(intent)
        status = str(res.get("status") or "")
        if mode == "webull" and status in ("submitted", "filled"):
            if act["side"] == "BUY":
                book.record_buy(
                    act["symbol"], act["quantity"], act["price"],
                    act["group"], act["fair_value"], act["z_score"], today_str,
                )
            else:
                book.record_sell(act["symbol"], act["price"], act["reason_code"], today_str)
            journal.append(
                {
                    "event": "fv_t_order",
                    "symbol": act["symbol"],
                    "side": act["side"],
                    "qty": act["quantity"],
                    "price": act["price"],
                    "fair_value": act["fair_value"],
                    "z_score": act["z_score"],
                    "reason": act["reason"],
                    "status": status,
                    "order": res.get("order"),
                }
            )
        executed.append({**act, "broker_status": status})

    return {"mode": mode, "regime": regime_ctx, "actions": actions, "executed": executed}


def watch_fv_t(
    settings: Settings,
    mode: str = "dry_run",
    confirm_live: bool = False,
    source: str = "yahoo",
    poll_minutes: int | None = None,
) -> None:
    """盘中定时轮询模式：常规时段每 poll_minutes 分钟刷新一次动态公允价并执行低买高卖。"""
    cfg = fv_t_cfg(settings)
    interval = max(1, int(poll_minutes or cfg.get("poll_minutes", 15)))
    print(f"[fv-t] 启动盘中轮询（每 {interval} 分钟一次，模式={mode}，Ctrl-C 退出）")
    first_round = True
    while True:
        now_et = datetime.now(ET)
        open_now, start_t, close_t = is_rth_open(now_et)
        if not open_now:
            print(f"[{now_et.strftime('%H:%M:%S')} ET] 当前非美股常规时段，等待 60 秒...")
            time.sleep(60)
            continue
        try:
            run_fv_t_once(
                settings=settings,
                mode=mode,
                confirm_live=confirm_live,
                source=source,
                no_fetch=not first_round,
                no_fund_fetch=not first_round,
                no_live=False,
            )
            first_round = False
        except Exception as e:
            print(f"[{now_et.strftime('%H:%M:%S')} ET] 本轮运行异常: {e}")
        time.sleep(interval * 60)


def print_fv_t_status(settings: Settings, mode: str = "webull") -> None:
    cfg = fv_t_cfg(settings)
    env_key = settings.environment if mode == "webull" else f"dry_{settings.environment}"
    book = FvTBook(env=env_key, capital=float(cfg["capital"]))
    print(f"=== fv-t 独立模拟账本（{book.path.name}） ===")
    print(
        f"初始资金: ${book.data['initial_capital']:,.2f} | 可用现金: ${book.cash:,.2f} | "
        f"累计已实现盈亏: ${book.data.get('realized_pnl', 0.0):+,.2f} | 更新时间: {book.data.get('updated_at') or '-'}"
    )
    if not book.positions:
        print("当前持仓: 空仓")
    else:
        print("当前持仓:")
        for sym, p in book.positions.items():
            print(
                f"  {sym:<5} ({p['group']:<7}) {p['qty']:>4d} 股 | 成本 ${p['cost_price']:.2f} | "
                f"入场日 {p['entry_date']} | 入场FV ${p['entry_fv']:.2f} (Z={p['entry_z']:+.2f})"
            )
    if book.trades_path.exists():
        lines = book.trades_path.read_text(encoding="utf-8").splitlines()
        print(f"\n最近已平仓做 T 记录（共 {len(lines)} 笔，展示最后 5 笔）:")
        for raw in lines[-5:]:
            t = json.loads(raw)
            print(
                f"  {t['exit_date']} {t['symbol']:<5} {t['qty']:>4d} 股 "
                f"${t['entry_price']:.2f} → ${t['exit_price']:.2f}  PnL ${t['pnl']:+7.2f} ({t['ret_pct']:+.2f}%) [{t['reason']}]"
            )
