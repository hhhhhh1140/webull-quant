"""Read-only full-position sizing from the trading account and order session."""
from __future__ import annotations

from decimal import Decimal

from .broker import build_webull_order
from . import futu_gateway
from .risk import OrderIntent
from .webull_client import WebullClient, _json, _unwrap_list


def webull_cash_budget(usd: dict, *, overnight: bool) -> Decimal:
    buying_power = Decimal(usd["buying_power"])
    if overnight:
        # The API returns an aggregate buying power, not a dedicated overnight cash figure.
        buying_power = min(buying_power, Decimal(usd["cash_balance"]), Decimal(usd["settled_cash"]))
    return max(Decimal(0), buying_power)


def plan_webull(client: WebullClient, account_suffix: str, symbol: str, limit: Decimal, session: str) -> dict:
    rows = _unwrap_list(_json(client.trade.account_v2.get_account_list()))
    ids = [str(r["account_id"]) for r in rows if str(r["account_id"]).endswith(account_suffix)]
    if len(ids) != 1:
        raise RuntimeError("Webull 账户尾号必须唯一匹配")
    account = ids[0]
    balance = _json(client.trade.account_v2.get_account_balance(account))
    usd = client._currency_row(balance)
    if usd is None:
        raise RuntimeError("Webull 账户缺少美元余额明细")
    overnight = session == "overnight"
    order_session = {"regular": "CORE", "extended": "ALL", "overnight": "ALL_DAY"}[session]
    budget = webull_cash_budget(usd, overnight=overnight)
    quantity = int(budget // limit)
    fees = Decimal(0)
    while quantity:
        order = build_webull_order(OrderIntent(symbol, "BUY", quantity, float(limit), "只读整股预算"),
                                   {"type": "LIMIT", "trading_session": "CORE", "limit_offset_bps": 0})
        order["support_trading_session"] = order_session
        preview = _json(client.trade.order_v3.preview_order(account_id=account, preview_orders=[order]))
        if isinstance(preview, list):
            preview = preview[0] if preview else {}
        if not isinstance(preview, dict) or "estimated_cost" not in preview:
            raise RuntimeError("Webull 预检没有提供预计订单成本")
        fees = Decimal(preview["estimated_transaction_fee"])
        if max(Decimal(preview["estimated_cost"]), quantity * limit) + fees <= budget:
            break
        quantity = max(0, min(quantity - 1, int((budget - fees) // limit)))
    return {"account": account_suffix, "session": order_session, "shares": quantity,
            "budget_usd": str(budget), "notional_usd": str(quantity * limit),
            "estimated_fee_usd": str(fees if quantity else 0),
            "basis": "usd_cash_estimate" if overnight else "reported_buying_power"}


def plan_futu(symbol: str, limit: Decimal, session: str) -> dict:
    sdk = futu_gateway._futu()
    order_session = {"regular": sdk.Session.RTH, "extended": sdk.Session.ETH,
                     "overnight": sdk.Session.OVERNIGHT}[session]
    context = futu_gateway._trade_context()
    try:
        account = futu_gateway.select_account(futu_gateway.us_accounts(context), "REAL")
        ret, rows = context.acctradinginfo_query(
            sdk.OrderType.NORMAL, f"US.{symbol}", float(limit), session=order_session,
            trd_env=sdk.TrdEnv.REAL, acc_id=int(account["acc_id"]),
        )
        if ret != sdk.RET_OK or rows.empty:
            raise RuntimeError(f"富途额度查询失败：{rows}")
        row = rows.iloc[0]
        quantity = max(0, int(row["max_cash_and_margin_buy"]))
        return {"account": str(account["acc_id"])[-4:], "session": order_session, "shares": quantity,
                "cash_only_shares": max(0, int(row["max_cash_buy"])),
                "notional_usd": str(quantity * limit), "basis": "cash_and_margin_quantity"}
    finally:
        context.close()
