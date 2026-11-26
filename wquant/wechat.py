"""Durable, read-only trade notifications for the SOXL/SOXS strategy."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Thread
from time import sleep, time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .config import ROOT, load_settings

log = logging.getLogger(__name__)
STATE = ROOT / "state"
SHANGHAI = ZoneInfo("Asia/Shanghai")
WPUSH_URL = "https://api.wpush.cn/api/v1/send"
PUSHPLUS_URL = "https://www.pushplus.plus/send"
PROVIDER_NAMES = {"pushplus": "PushPlus", "wpush": "WPush"}
BROKER_STATES = {
    "PARTIAL_FILLED": "部分成交", "FILLED": "全部成交", "FAILED": "券商拒单", "REJECTED": "券商拒单",
    "CANCELED": "已撤单", "CANCELLED": "已撤单", "EXPIRED": "已过期",
}


class PushError(RuntimeError):
    pass


def push(title: str, body: str, token: str, *, provider: str = "wpush") -> str:
    payload = {"title": title, "content": body, "channel": "wechat"}
    if provider == "pushplus":
        data = json.dumps({**payload, "token": token, "template": "txt"}, ensure_ascii=False).encode()
        request = Request(PUSHPLUS_URL, data=data, headers={"Content-Type": "application/json"}, method="POST")
        success_code = 200
    else:
        data = urlencode({**payload, "apikey": token}).encode()
        request = Request(WPUSH_URL, data=data, method="POST")
        success_code = 0
    with urlopen(request, timeout=5) as response:
        result = json.load(response)
    if result.get("code") != success_code or not result.get("data"):
        code = result.get("code")
        name = PROVIDER_NAMES[provider]
        raise PushError(f"{name} 拒绝请求（代码 {code if isinstance(code, int) else '未知'}）")
    return str(result["data"])


def token_path() -> Path:
    return STATE / "wechat_token"


def read_channel() -> tuple[str, str]:
    try:
        saved = token_path().read_text().strip()
    except FileNotFoundError:
        return "", ""
    if saved.startswith("{"):
        channel = json.loads(saved)
        return channel["provider"], channel["token"]
    return "wpush", saved


def save_channel(provider: str, token: str) -> None:
    path = token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as temporary:
        json.dump({"provider": provider, "token": token}, temporary)
    try:
        os.replace(temporary.name, path)
    finally:
        Path(temporary.name).unlink(missing_ok=True)


@contextmanager
def _db():
    STATE.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(STATE / "wechat.sqlite3", timeout=1.0)) as db, db:
        db.row_factory = sqlite3.Row
        db.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY, title TEXT, body TEXT, next_try REAL, failures INTEGER DEFAULT 0
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS pending_orders (
            environment TEXT, account_id TEXT, client_id TEXT, payload TEXT,
            last_fill TEXT DEFAULT '', next_poll REAL, expires_at REAL,
            PRIMARY KEY (environment, account_id, client_id)
        )""")
        yield db


def _enqueue(db, payload, status, *, filled_qty=None, filled_price=None):
    side = {"BUY": "买入", "SELL": "卖出"}[payload["side"]]
    title = f"{payload['account']}｜{payload['symbol']} {side} · {status}"
    body = [f"北京时间 {datetime.now(SHANGHAI):%Y-%m-%d %H:%M:%S}",
            f"数量 {payload['qty']} 股 · 触发报价 USD {payload['quote']:.2f}"]
    if payload.get("limit") is not None:
        body.append(f"委托限价 USD {payload['limit']}")
    if payload.get("stop") is not None:
        body.append(f"止损触发线 USD {payload['stop']:.2f}")
    if filled_qty not in (None, "", "0", 0):
        fill = f"累计成交 {filled_qty} 股"
        if filled_price not in (None, "", "0", 0):
            fill += f" · 成交均价 USD {filled_price}"
        body.append(fill)
    if payload.get("session"):
        body.append(f"委托时段 {payload['session']}")
    body.append(f"原因 {payload['reason']}")
    db.execute("INSERT INTO messages (title, body, next_try) VALUES (?, ?, ?)", (title, "\n".join(body), time()))


def record_order(broker, intent, result, *, stop_price=None, include_initial=True):
    """Record strategy orders locally. Never send requests to the notification service from the strategy."""
    name = getattr(broker, "name", None)
    if name == "webull":
        environment = broker.settings.environment
        account = "Webull 实盘" if environment == "prod" else "Webull 模拟盘"
    elif name == "futu-shadow":
        account = "富途影子盘"
    else:
        return

    order = result.get("order") or {}
    status = result.get("status")
    label = {"submitted": "已报单", "dry_run": "影子报价", "filled": "模拟成交",
             "error": "报单异常，请核对", "rejected": "拒单"}.get(status, "订单状态待核对")
    session = order.get("support_trading_session") or order.get("session")
    payload = {"account": account, "symbol": intent.symbol, "side": intent.side, "qty": intent.quantity,
               "quote": intent.price, "reason": intent.reason, "stop": stop_price,
               "limit": order.get("limit_price"),
               "session": {"ALL_DAY": "夜盘", "OVERNIGHT": "夜盘", "NIGHT": "夜盘", "CORE": "常规盘",
                           "RTH": "常规盘", "ALL": "扩展时段委托", "ETH": "扩展时段委托"}.get(session)}
    if stop_price is not None:
        label = f"止损触发 · {label}"
    with _db() as db:
        if include_initial:
            _enqueue(db, payload, label, filled_qty=result.get("quantity"), filled_price=result.get("price"))
        if name == "webull" and order.get("client_order_id") and (status == "submitted" or result.get("attempted")):
            db.execute("""INSERT OR IGNORE INTO pending_orders
                (environment, account_id, client_id, payload, next_poll, expires_at) VALUES (?, ?, ?, ?, ?, ?)""",
                (environment, broker.client.account_id, order["client_order_id"], json.dumps(payload),
                 time(), time() + 7 * 86400))


def _send_pending(token: str, *, send=push):
    with _db() as db:
        due = db.execute("SELECT * FROM messages WHERE next_try <= ? ORDER BY id LIMIT 10", (time(),)).fetchall()
        for row in due:
            try:
                send(row["title"], row["body"], token)
            except (OSError, ValueError, PushError) as exc:
                failures = row["failures"] + 1
                delay = min(300, 2 ** min(failures, 9))
                log.warning("微信通知提交失败（消息 %s，%s），%s 秒后重试", row["id"], type(exc).__name__, delay)
                db.execute("UPDATE messages SET next_try = ?, failures = ? WHERE id = ?",
                           (time() + delay, failures, row["id"]))
                db.commit()
                return
            db.execute("DELETE FROM messages WHERE id = ?", (row["id"],))
            db.commit()


def _poll_pending(lookup):
    with _db() as db:
        db.execute("DELETE FROM pending_orders WHERE expires_at <= ?", (time(),))
        db.commit()
        due = db.execute("SELECT * FROM pending_orders WHERE next_poll <= ? ORDER BY next_poll LIMIT 20",
                         (time(),)).fetchall()
        for row in due:
            key = (row["environment"], row["account_id"], row["client_id"])
            where = " WHERE environment = ? AND account_id = ? AND client_id = ?"
            try:
                detail = lookup(*key)
            except Exception as exc:
                log.warning("Webull %s 订单状态读取失败（%s），稍后重试", row["environment"], type(exc).__name__)
                db.execute("UPDATE pending_orders SET next_poll = ?" + where, (time() + 60, *key))
                db.commit()
                continue
            status = str(detail.get("status", "")).upper()
            filled = str(detail.get("filled_quantity") or "")
            if status in BROKER_STATES and (status != "PARTIAL_FILLED" or filled != row["last_fill"]):
                payload = json.loads(row["payload"])
                _enqueue(db, payload, BROKER_STATES[status], filled_qty=filled,
                         filled_price=detail.get("filled_price"))
            if status in BROKER_STATES and status != "PARTIAL_FILLED":
                db.execute("DELETE FROM pending_orders" + where, key)
            else:
                db.execute("UPDATE pending_orders SET last_fill = ?, next_poll = ?" + where,
                           (filled or row["last_fill"], time() + 3, *key))
            db.commit()


def counts() -> tuple[int, int]:
    with _db() as db:
        messages = db.execute("SELECT count(*) FROM messages").fetchone()[0]
        orders = db.execute("SELECT count(*) FROM pending_orders").fetchone()[0]
    return messages, orders


def watch(config_path=None):
    """Send the outbox and query Webull by client ID using read-only API calls."""
    from fcntl import LOCK_EX, LOCK_NB, flock

    from . import clawbot
    from .webull_client import WebullClient

    clients = {}

    def lookup(environment, account_id, client_id):
        if environment not in clients:
            clients[environment] = WebullClient(load_settings(config_path, environment=environment))
        return clients[environment].order_detail(client_id, account_id=account_id)

    def poll():
        while True:
            try:
                _poll_pending(lookup)
            except sqlite3.Error as exc:
                log.warning("订单通知队列暂时不可用（%s），稍后重试", type(exc).__name__)
            sleep(1)

    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "wechat.lock").open("a") as lock:
        try:
            flock(lock, LOCK_EX | LOCK_NB)
        except BlockingIOError:
            raise SystemExit("微信通知进程已在运行") from None
        print("微信通知进程已启动，等待绑定" if not read_channel()[1] else "微信通知进程已启动", flush=True)
        tracker = Thread(target=poll, daemon=True)
        tracker.start()
        inbox = Thread(target=clawbot.watch, args=(config_path,), daemon=True)
        inbox.start()
        while True:
            try:
                account, sync = clawbot.state()
                if account:
                    if sync.get("context"):
                        _send_pending("", send=clawbot.push)
                else:
                    provider, token = read_channel()
                    if token:
                        _send_pending(token, send=lambda title, body, token: push(title, body, token, provider=provider))
            except (sqlite3.Error, OSError, ValueError) as exc:
                log.warning("微信通知队列暂时不可用（%s），稍后重试", type(exc).__name__)
            if not tracker.is_alive() or not inbox.is_alive():
                raise RuntimeError("Webull 订单或微信消息监控已退出")
            sleep(1)
