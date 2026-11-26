from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from wquant import futu_gateway
from wquant.broker import build_webull_order
from wquant.risk import OrderIntent


def test_shadow_selects_us_account_and_builds_overnight_limit(monkeypatch):
    class AccountsOnly:
        def get_acc_list(self):
            return 0, pd.DataFrame([
                {"acc_id": 10000001, "acc_status": "ACTIVE", "trd_env": "REAL", "trdmarket_auth": ["US"]},
                {"acc_id": 10000002, "acc_status": "ACTIVE", "trd_env": "SIMULATE", "trdmarket_auth": ["US"]},
            ])

        def close(self):
            pass

    class Futu:
        RET_OK = 0

    now = datetime(2026, 11, 23, 21, 30, tzinfo=ZoneInfo("America/New_York"))
    monkeypatch.setattr(futu_gateway, "_futu", lambda: Futu)
    monkeypatch.setattr(futu_gateway, "_trade_context", AccountsOnly)
    monkeypatch.setattr(futu_gateway, "build_webull_order",
                        lambda intent, config: build_webull_order(intent, config, now))
    broker = futu_gateway.FutuShadowBroker({"trading_session": "ALL_DAY", "limit_offset_bps": 30})
    result = broker.submit(OrderIntent("SOXL", "BUY", 7, 100.00, "shadow"))

    assert broker.book_environment.startswith("futu_real_shadow_")
    assert result["status"] == "dry_run"
    assert result["order"] == {
        "symbol": "US.SOXL", "side": "BUY", "quantity": 7, "limit_price": "100.30",
        "order_type": "NORMAL", "session": "ALL", "time_in_force": "DAY",
        "trd_env": "REAL", "account_suffix": "0001",
    }
