"""执行层：把风控批准的 OrderIntent 变成真实订单。

三种模式:
- DryRunBroker: 只打印
- PaperBroker:  本地 JSON 文件模拟账户，按参考价成交
- WebullBroker: 先 preview 再 place；环境（uat/prod）由 .env 决定
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from .config import ROOT, Settings
from .risk import OrderIntent


def session_for(configured: str, now=None) -> str:
    """Webull 只允许在夜盘（20:00–04:00 ET）下 ALL_DAY 单，其他时间会被拒（OPENAPI_ALL_DAY_ONLY_ALLOW_IN_NIGHT_TRADING_TIME_RANGE）。
    配置成 ALL_DAY 时：夜盘用 ALL_DAY，04:00–20:00 改用 ALL（盘前 + 常规 + 盘后）。"""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    s = configured.upper()
    if s != "ALL_DAY":
        return s
    t = (now or datetime.now(ZoneInfo("America/New_York"))).astimezone(ZoneInfo("America/New_York"))
    return "ALL_DAY" if (t.hour >= 20 or t.hour < 4) else "ALL"


def build_webull_order(intent: OrderIntent, order_cfg: dict, now=None) -> dict:
    """订单格式与官方 webull-openapi-mcp 的 _build_stock_order 一致。"""
    otype = order_cfg.get("type", "LIMIT").upper()
    session = session_for(order_cfg.get("trading_session", "CORE"), now)
    if session != "CORE":
        otype = "LIMIT"   # 盘前盘后/夜盘流动性差，只允许限价单
    order = {
        "combo_type": "NORMAL",
        "client_order_id": uuid.uuid4().hex[:32],
        "instrument_type": "EQUITY",
        "market": "US",
        "symbol": intent.symbol,
        "order_type": otype,
        "entrust_type": "QTY",
        "support_trading_session": session,
        "time_in_force": order_cfg.get("time_in_force", "DAY"),
        "side": intent.side,
        "quantity": str(int(intent.quantity)),
    }
    if otype == "LIMIT":
        off = order_cfg.get("limit_offset_bps", 20) / 1e4
        px = intent.price * (1 + off) if intent.side == "BUY" else intent.price * (1 - off)
        order["limit_price"] = f"{px:.2f}"
    return order


class DryRunBroker:
    name = "dry_run"

    def __init__(self, order_cfg: dict):
        self.order_cfg = order_cfg

    def submit(self, intent: OrderIntent) -> dict:
        order = build_webull_order(intent, self.order_cfg)
        print(f"  [DRY RUN] 不会下单: {json.dumps(order, ensure_ascii=False)}")
        return {"status": "dry_run", "order": order}


class PaperBroker:
    """本地模拟账户，状态存在 state/paper_account.json。"""
    name = "paper"

    def __init__(self, order_cfg: dict, initial_cash: float = 100_000, path: Path | None = None):
        self.order_cfg = order_cfg
        self.path = path or ROOT / "state" / "paper_account.json"
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"cash": initial_cash, "positions": {}}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2))

    def cash(self) -> float:
        return float(self.state["cash"])

    def positions(self) -> dict[str, dict]:
        return self.state["positions"]

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash() + sum(prices.get(s, p["cost_price"]) * p["quantity"] for s, p in self.positions().items())

    def submit(self, intent: OrderIntent) -> dict:
        pos = self.state["positions"]
        if intent.side == "BUY":
            cost = intent.price * intent.quantity
            if cost > self.state["cash"]:
                return {"status": "rejected", "reason": "现金不足"}
            self.state["cash"] -= cost
            p = pos.setdefault(intent.symbol, {"quantity": 0, "cost_price": 0.0, "high_close": intent.price})
            total = p["quantity"] + intent.quantity
            p["cost_price"] = (p["cost_price"] * p["quantity"] + cost) / total
            p["quantity"] = total
        else:
            p = pos.get(intent.symbol)
            if not p:
                return {"status": "rejected", "reason": "无持仓"}
            q = min(intent.quantity, p["quantity"])
            self.state["cash"] += intent.price * q
            p["quantity"] -= q
            if p["quantity"] <= 0:
                pos.pop(intent.symbol)
        self._save()
        return {"status": "filled", "price": intent.price, "quantity": intent.quantity}


class WebullBroker:
    name = "webull"

    def __init__(self, settings: Settings, client, order_cfg: dict, confirm_live: bool = False):
        if settings.environment == "prod":
            if not settings.live_trading_enabled:
                raise RuntimeError("WEBULL_ENVIRONMENT=prod，但 config.yaml 的 live_trading_enabled 不是 true，拒绝实盘下单")
            if not confirm_live:
                raise RuntimeError("实盘下单需要在命令行加 --confirm-live")
        self.settings = settings
        self.client = client
        self.order_cfg = order_cfg

    def submit(self, intent: OrderIntent) -> dict:
        order = build_webull_order(intent, self.order_cfg)
        preview = self.client.preview_order(order)
        print(f"  [{self.settings.environment.upper()}] preview: {json.dumps(preview, ensure_ascii=False, default=str)[:300]}")
        result = self.client.place_order(order)
        print(f"  [{self.settings.environment.upper()}] placed: {json.dumps(result, ensure_ascii=False, default=str)[:300]}")
        return {"status": "submitted", "order": order, "preview": preview, "result": result}
