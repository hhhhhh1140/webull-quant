from decimal import Decimal
from types import SimpleNamespace

from wquant.full_buy_plan import plan_webull, webull_cash_budget


def test_webull_overnight_budget_uses_cash_and_settlement():
    usd = {"cash_balance": "352.09", "settled_cash": "443.05", "buying_power": "2988.18"}
    assert webull_cash_budget(usd, overnight=True) == Decimal("352.09")
    assert webull_cash_budget(usd, overnight=False) == Decimal("2988.18")


def test_webull_full_buy_plan_rounds_down_after_fees():
    attempts = []

    def preview_order(*, account_id, preview_orders):
        order = preview_orders[0]
        assert account_id == "fake-1234"
        assert order["support_trading_session"] == "ALL_DAY"
        quantity = int(order["quantity"])
        attempts.append(quantity)
        return {"estimated_cost": str(quantity * 20), "estimated_transaction_fee": "1.00"}

    usd = {"cash_balance": "100.00", "settled_cash": "100.00", "buying_power": "200.00"}
    trade = SimpleNamespace(
        account_v2=SimpleNamespace(get_account_list=lambda: [{"account_id": "fake-1234"}],
                                   get_account_balance=lambda account: usd),
        order_v3=SimpleNamespace(preview_order=preview_order),
    )
    client = SimpleNamespace(trade=trade, _currency_row=lambda balance: balance)
    plan = plan_webull(client, "1234", "SOXL", Decimal("20.00"), "overnight")
    assert attempts == [5, 4]
    assert plan["shares"] == 4 and plan["basis"] == "usd_cash_estimate"
    assert plan["notional_usd"] == "80.00"
