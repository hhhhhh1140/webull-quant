"""Private WeChat chat using the iLink API used by Tencent's ClawBot plugin."""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from secrets import randbits, token_hex
from tempfile import NamedTemporaryFile
from time import monotonic, sleep, time
from urllib.parse import urlsplit

import requests

from .config import ROOT

log = logging.getLogger(__name__)
STATE = ROOT / "state"
WEIXIN = "https://ilinkai.weixin.qq.com"
VERSION = "2.4.10"


class ClawError(RuntimeError):
    pass


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as tmp:
        json.dump(data, tmp, ensure_ascii=False)
    try:
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def openai_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip() or _read(STATE / "clawbot-openai.json").get("key", "")


def save_openai_key(key: str) -> None:
    _save(STATE / "clawbot-openai.json", {"key": key})


def gemini_key() -> str:
    return (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
        or _read(STATE / "clawbot-gemini.json").get("key", "")
    )


def save_gemini_key(key: str) -> None:
    _save(STATE / "clawbot-gemini.json", {"key": key})


def chat_provider() -> str:
    val = (os.environ.get("WQUANT_CHAT_PROVIDER") or _read(STATE / "clawbot-provider.json").get("provider") or "auto").strip().lower()
    return val if val in ("auto", "gemini", "codex", "openai") else "auto"


def save_chat_provider(provider: str) -> None:
    if provider not in ("auto", "gemini", "codex", "openai"):
        raise ValueError(f"不支持的对话模型引擎：{provider}")
    _save(STATE / "clawbot-provider.json", {"provider": provider})


def _host(url: str) -> str:
    base = urlsplit(url)
    if (base.scheme != "https" or not (base.hostname or "").endswith(".weixin.qq.com") or base.port not in (None, 443)
            or base.username or base.password or base.path not in ("", "/") or base.query or base.fragment):
        raise ClawError("微信返回了无法识别的连接地址")
    return f"https://{base.hostname}"


def _request(endpoint: str, *, server=WEIXIN, token=None, data=None, params=None, timeout=12) -> dict:
    headers = {"iLink-App-Id": "bot", "iLink-App-ClientVersion": str((2 << 16) | (4 << 8) | 10)}
    if data is not None:
        headers["AuthorizationType"] = "ilink_bot_token"
        headers["X-WECHAT-UIN"] = base64.b64encode(str(randbits(32)).encode()).decode()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        data = {**data, "base_info": {"channel_version": VERSION, "bot_agent": "WebullQuant/1.0"}}
    response = requests.request("POST" if data is not None else "GET", f"{_host(server)}/ilink/bot/{endpoint}",
                                headers=headers, params=params, json=data, timeout=timeout, allow_redirects=False)
    response.raise_for_status()
    result = response.json()
    if result.get("errcode") or result.get("ret"):
        raise ClawError(f"微信接口返回错误码 {result.get('errcode') or result['ret']}")
    return result


def pairing() -> None:
    from getpass import getpass

    import qrcode

    login = _request("get_bot_qrcode", data={"local_token_list": []}, params={"bot_type": "3"})
    qr = qrcode.QRCode(border=2)
    qr.add_data(login["qrcode_img_content"])
    print("请用手机微信扫一扫，确认连接微信 ClawBot。", flush=True)
    qr.print_ascii(tty=True)
    server, verify = WEIXIN, ""
    deadline = monotonic() + 300
    while monotonic() < deadline:
        params = {"qrcode": login["qrcode"]}
        if verify:
            params["verify_code"] = verify
        try:
            status = _request("get_qrcode_status", server=server, params=params, timeout=40)
        except requests.Timeout:
            continue
        verify = ""
        match status.get("status"):
            case "scaned_but_redirect":
                server = _host(f"https://{status['redirect_host']}")
            case "need_verifycode":
                verify = getpass("输入手机微信显示的验证码：").strip()
            case "confirmed":
                token, owner = status.get("bot_token"), status.get("ilink_user_id")
                if not token or not owner:
                    raise ClawError("微信未返回机器人凭证或扫码人的身份")
                _save(STATE / "clawbot.json", {"token": token, "owner": owner,
                                               "server": _host(status.get("baseurl") or server)})
                print("微信已绑定。请在新开的 ClawBot 对话里发「帮助」验证收发。", flush=True)
                return
            case "expired" | "verify_code_blocked" | "binded_redirect":
                raise ClawError("微信未发放新的登录凭证，请在微信中检查绑定状态后重试")
        sleep(0.5)
    raise ClawError("二维码已超时，请重新启动绑定")


def _key(account: dict) -> str:
    from hashlib import sha256

    return sha256(account["token"].encode()).hexdigest()[:24]


def _sync_path(account: dict) -> Path:
    return STATE / f"clawbot-sync-{_key(account)}.json"


def state() -> tuple[dict, dict]:
    account = _read(STATE / "clawbot.json")
    return account, _read(_sync_path(account)) if account else {}


def _send(account: dict, context: str, text: str) -> str:
    response = _request("sendmessage", server=account["server"], token=account["token"], data={"msg": {
        "from_user_id": "", "to_user_id": account["owner"], "client_id": "wquant-" + token_hex(12),
        "context_token": context, "message_type": 2, "message_state": 2,
        "item_list": [{"type": 1, "text_item": {"text": text[:3500]}}],
    }})
    return str(response.get("message_id", ""))


def push(title: str, body: str, _token: str = "") -> str:
    from .wechat import PushError

    account, sync = state()
    if not account or not sync.get("context"):
        raise PushError("需要在微信 ClawBot 对话里发一条消息来激活会话")
    if time() - sync.get("active_at", 0) >= 24 * 3600:
        raise PushError("微信 ClawBot 会话窗口已到期")
    try:
        return _send(account, sync["context"], f"{title}\n{body}")
    except ClawError as exc:
        raise PushError(str(exc)) from exc


def handle(account: dict, sync: dict, messages: list, chat) -> None:
    for message in messages:
        if (message.get("from_user_id") != account["owner"] or message.get("group_id")
                or message.get("message_type") != 1):
            continue
        identity = message.get("message_id") or message.get("client_id") or message.get("seq")
        context = message.get("context_token")
        if not identity or not context or str(identity) in sync.get("seen", ()):
            continue
        text = "\n".join(str((item.get("text_item") or {}).get("text", ""))
                         for item in message.get("item_list", ()) if item.get("type") == 1).strip()
        if not text:
            continue
        # Persist before a confirmation can reach the broker, so redelivery cannot replay it.
        sync["seen"] = [*sync.get("seen", ()), str(identity)][-500:]
        sync["context"], sync["active_at"] = context, time()
        _save(_sync_path(account), sync)
        try:
            answer = chat.respond(text)
        except Exception as exc:
            log.warning("微信会话命令暂时失败（%s）", type(exc).__name__)
            answer = "这条请求暂时处理不了。若刚才确认了订单，请先核对券商订单，再决定是否重试。"
        _send(account, context, answer)
        sync["last_reply_at"] = time()
        _save(_sync_path(account), sync)


def watch(config_path=None) -> None:
    from .chat_trade import Chat

    current_key = ""
    chat = None
    while True:
        try:
            account, sync = state()
            if not account:
                sleep(2)
                continue
            if current_key != _key(account):
                chat = Chat(config_path)
                current_key = _key(account)
            batch = _request("getupdates", server=account["server"], token=account["token"],
                             data={"get_updates_buf": sync.get("cursor", "")}, timeout=40)
            handle(account, sync, batch.get("msgs", []), chat)
            cursor = batch.get("get_updates_buf") or sync.get("cursor", "")
            if cursor != sync.get("cursor", "") or time() - sync.get("last_poll_at", 0) >= 60:
                sync["cursor"] = cursor
                sync["last_poll_at"] = time()
                _save(_sync_path(account), sync)
        except requests.Timeout:
            continue
        except (OSError, ValueError, ClawError) as exc:
            log.warning("微信 ClawBot 暂时无法收发（%s），稍后重试", type(exc).__name__)
            sleep(8)
