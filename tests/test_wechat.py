from io import BytesIO
import json
from types import SimpleNamespace
from urllib.parse import parse_qs

import pytest

from wquant import swing_live, wechat
from wquant.risk import OrderIntent
from wquant.webull_client import WebullClient


@pytest.fixture(autouse=True)
def local_state(monkeypatch, tmp_path):
    monkeypatch.setattr(wechat, "STATE", tmp_path)


def test_stop_order_alert_and_cumulative_fills_survive_worker_restart(monkeypatch):
    broker = SimpleNamespace(name="webull", settings=SimpleNamespace(environment="prod"),
                             client=SimpleNamespace(account_id="private-account-id"))
    intent = OrderIntent("SOXS", "SELL", 20, 60.15, "移动止损")
    order = {"client_order_id": "private-order-id", "limit_price": "60.03", "support_trading_session": "ALL_DAY"}
    clock = [1000]
    monkeypatch.setattr(wechat, "time", lambda: clock[0])
    wechat.record_order(broker, intent, {"status": "submitted", "order": order}, stop_price=60.72)

    status = {"status": "PARTIAL_FILLED", "filled_quantity": "5", "filled_price": "60.10"}

    def lookup(environment, account_id, client_id):
        assert (environment, account_id, client_id) == ("prod", "private-account-id", "private-order-id")
        return status

    for _ in range(2):
        wechat._poll_pending(lookup)
        clock[0] += 4
    status.update(status="FILLED", filled_quantity="20", filled_price="60.08")
    wechat._poll_pending(lookup)
    assert wechat.counts() == (3, 0)

    sent = []
    wechat._send_pending("token", send=lambda title, body, token: sent.append((title, body)))
    assert wechat.counts() == (0, 0)
    assert [title for title, _ in sent] == [
        "Webull 实盘｜SOXS 卖出 · 止损触发 · 已报单",
        "Webull 实盘｜SOXS 卖出 · 部分成交",
        "Webull 实盘｜SOXS 卖出 · 全部成交",
    ]
    assert "报价 USD 60.15" in sent[0][1] and "限价 USD 60.03" in sent[0][1]
    assert "止损触发线 USD 60.72" in sent[0][1] and "委托时段 夜盘" in sent[0][1]
    assert "累计成交 20 股 · 成交均价 USD 60.08" in sent[-1][1]
    assert all("private-" not in body for _, body in sent)


def test_provider_failure_retries_futu_shadow_without_tracking_order(monkeypatch):
    clock = [1000]
    monkeypatch.setattr(wechat, "time", lambda: clock[0])
    intent = OrderIntent("SOXL", "BUY", 5, 35, "入场")
    wechat.record_order(SimpleNamespace(name="dry_run"), intent, {"status": "dry_run"})
    wechat.record_order(SimpleNamespace(name="futu-shadow"), intent, {"status": "dry_run", "order": {"limit_price": "35.07"}})

    def offline(*args):
        raise TimeoutError()

    wechat._send_pending("token", send=offline)
    assert wechat.counts() == (1, 0)
    clock[0] += 3
    sent = []
    wechat._send_pending("token", send=lambda title, *args: sent.append(title))
    assert sent == ["富途影子盘｜SOXL 买入 · 影子报价"]
    assert wechat.counts() == (0, 0)


def test_alert_failure_does_not_change_order_outcome(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(wechat, "record_order", fail)
    monkeypatch.setattr(swing_live.journal, "record", lambda *args, **kwargs: None)
    broker = SimpleNamespace(submit=lambda intent: {"status": "submitted"})
    assert swing_live._submit(broker, "SOXL", "BUY", 5, 35, "入场")


def test_push_submits_only_to_the_bound_wechat_account(monkeypatch):
    def open_request(request, timeout):
        assert request.full_url == "https://api.wpush.cn/api/v1/send"
        assert request.method == "POST"
        assert parse_qs(request.data.decode()) == {"apikey": ["key"], "title": ["标题"],
                                                  "content": ["内容"], "channel": ["wechat"]}
        return BytesIO(b'{"code":0,"data":"receipt"}')

    monkeypatch.setattr(wechat, "urlopen", open_request)
    assert wechat.push("标题", "内容", "key") == "receipt"


def test_pushplus_uses_a_private_wechat_message(monkeypatch):
    def open_request(request, timeout):
        assert request.full_url == "https://www.pushplus.plus/send"
        assert request.method == "POST"
        assert json.loads(request.data) == {"token": "key", "title": "标题", "content": "内容",
                                           "channel": "wechat", "template": "txt"}
        return BytesIO(b'{"code":200,"data":"receipt"}')

    monkeypatch.setattr(wechat, "urlopen", open_request)
    assert wechat.push("标题", "内容", "key", provider="pushplus") == "receipt"


def test_read_channel_preserves_existing_wpush_binding():
    wechat.token_path().write_text("old-key\n")
    assert wechat.read_channel() == ("wpush", "old-key")
    wechat.save_channel("pushplus", "new-key")
    assert wechat.read_channel() == ("pushplus", "new-key")
    assert wechat.token_path().stat().st_mode & 0o777 == 0o600


def test_order_details_match_exact_client_id():
    def get_order_detail(account, client_id):
        assert (account, client_id) == ("acct", "ours")
        return {"data": {"orders": [{"client_order_id": "other", "status": "CANCELED"},
                                    {"client_order_id": "ours", "status": "FILLED"}]}}

    client = WebullClient.__new__(WebullClient)
    client.trade = SimpleNamespace(order_v3=SimpleNamespace(get_order_detail=get_order_detail))
    assert client.order_detail("ours", account_id="acct")["status"] == "FILLED"
