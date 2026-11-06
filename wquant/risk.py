"""风控层：所有订单在下单前都必须经过这里。回测和实盘共用同一套规则。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import ROOT

KILL_SWITCH = ROOT / "KILL_SWITCH"


@dataclass
class OrderIntent:
    symbol: str
    side: str            # BUY | SELL
    quantity: float
    price: float         # 参考价
    reason: str = ""

    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass
class RiskDecision:
    approved: bool
    intent: OrderIntent | None
    notes: list[str] = field(default_factory=list)


class RiskManager:
    def __init__(self, cfg: dict, whitelist: list[str]):
        self.cfg = cfg
        self.whitelist = {s.upper() for s in whitelist}

    @staticmethod
    def kill_switch_on() -> bool:
        return KILL_SWITCH.exists()

    def daily_loss_breached(self, equity: float, start_of_day_equity: float | None) -> bool:
        if not start_of_day_equity:
            return False
        return (equity / start_of_day_equity - 1) <= -self.cfg.get("max_daily_loss_pct", 0.03)

    def size_buy(self, symbol: str, price: float, equity: float, cash: float, n_positions: int,
                 conviction: float = 1.0, start_of_day_equity: float | None = None,
                 leverage: float = 1.0, group: str | None = None, group_count: int = 0,
                 fractional: bool = False) -> RiskDecision:
        """leverage：杠杆 ETF 的倍数，目标仓位和单笔上限都按倍数缩小，保持风险敞口一致。
        group / group_count：所属板块和该板块已有持仓数，用于 max_per_group 限制。"""
        notes: list[str] = []
        if self.kill_switch_on():
            return RiskDecision(False, None, ["KILL_SWITCH 文件存在，停止一切交易"])
        if symbol.upper() not in self.whitelist:
            return RiskDecision(False, None, [f"{symbol} 不在白名单"])
        if n_positions >= self.cfg.get("max_positions", 6):
            return RiskDecision(False, None, ["已达最大持仓数"])
        max_g = self.cfg.get("max_per_group")
        if max_g and group and group != "default" and group_count >= max_g:
            return RiskDecision(False, None, [f"板块 {group} 已有 {group_count} 只持仓，达到上限 {max_g}"])
        if self.daily_loss_breached(equity, start_of_day_equity):
            return RiskDecision(False, None, ["触发当日最大亏损，停止开新仓"])
        if price <= 0:
            return RiskDecision(False, None, ["价格无效"])

        conviction = min(max(conviction, 0.0), 1.0)
        lev = max(float(leverage or 1), 1.0)
        target = equity * self.cfg.get("max_position_pct", 0.10) * conviction / lev
        spendable = cash - equity * self.cfg.get("cash_buffer_pct", 0.05)
        cap = self.cfg.get("max_order_notional", 5000) / lev
        if lev > 1:
            notes.append(f"{lev:g} 倍杠杆，仓位按 1/{lev:g} 计算")
        notional = min(target, spendable, cap)
        if notional < target:
            notes.append(f"金额由 {target:.0f} 限制为 {notional:.0f}")
        if notional < self.cfg.get("min_order_notional", 100):
            return RiskDecision(False, None, notes + ["可用金额太小"])
        # 回测用复权价格时，经历多次合股的 ETF 历史价格可能高达数万美元，按整股算会买不起，所以回测允许零碎股
        qty = round(notional / price, 6) if fractional else math.floor(notional / price)
        if qty <= 0 or (not fractional and qty < 1):
            return RiskDecision(False, None, notes + ["不足 1 股"])
        return RiskDecision(True, OrderIntent(symbol, "BUY", qty, price), notes)

    def size_hedge(self, symbol: str, price: float, equity: float, cash: float, n_positions: int,
                   conviction: float, max_pct: float, leverage: float = 1.0,
                   start_of_day_equity: float | None = None) -> RiskDecision:
        """对冲仓位：金额 = 本金 × max_pct × conviction，单笔上限按杠杆缩小。"""
        if self.kill_switch_on():
            return RiskDecision(False, None, ["KILL_SWITCH 文件存在，停止一切交易"])
        if symbol.upper() not in self.whitelist:
            return RiskDecision(False, None, [f"{symbol} 不在白名单"])
        if n_positions >= self.cfg.get("max_positions", 6):
            return RiskDecision(False, None, ["已达最大持仓数"])
        if self.daily_loss_breached(equity, start_of_day_equity):
            return RiskDecision(False, None, ["触发当日最大亏损，停止开新仓"])
        if price <= 0:
            return RiskDecision(False, None, ["价格无效"])
        lev = max(float(leverage or 1), 1.0)
        target = equity * max_pct * min(max(conviction, 0.0), 1.0)
        spendable = cash - equity * self.cfg.get("cash_buffer_pct", 0.05)
        notional = min(target, spendable, self.cfg.get("max_order_notional", 5000) / lev)
        notes = [f"对冲仓位：本金 × {max_pct:.0%} × {conviction:.2f}"]
        if notional < self.cfg.get("min_order_notional", 100):
            return RiskDecision(False, None, notes + ["可用金额太小"])
        qty = math.floor(notional / price)
        if qty < 1:
            return RiskDecision(False, None, notes + ["不足 1 股"])
        return RiskDecision(True, OrderIntent(symbol, "BUY", qty, price), notes)

    def check_sell(self, symbol: str, held_qty: float, price: float) -> RiskDecision:
        # 卖出（减仓/止损）不受 kill switch 以外的限制；kill switch 下也允许平仓
        if held_qty <= 0:
            return RiskDecision(False, None, ["无持仓"])
        qty = int(held_qty)
        cap = self.cfg.get("max_order_notional", 5000)
        notes = []
        if qty * price > cap * 4:
            notes.append("卖出金额较大，请留意")
        return RiskDecision(True, OrderIntent(symbol, "SELL", qty, price), notes)
