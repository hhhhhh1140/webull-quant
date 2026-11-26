"""Owner-only chat commands. GPT has no access to the broker or order confirmations."""
from __future__ import annotations

import logging
import os
import re
import subprocess
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from secrets import token_hex
from time import monotonic
from zoneinfo import ZoneInfo

import requests

from . import chat_strategy, codex_chat, gemini_chat, journal, wechat
from .broker import build_webull_order, session_for
from .clawbot import chat_provider, gemini_key, openai_key, save_chat_provider
from .config import load_settings
from .risk import OrderIntent
from .webull_client import WebullClient

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
ORDER = re.compile(
    r"(?:/下单\s*)?(?:Webull\s*)?(?:模拟盘\s*)?(买入|卖出)\s*(SOXL|SOXS)\s*(\d{1,6})\s*股\s*(?:限价\s*|@\s*)\$?(\d{1,5}(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
CONFIRM = re.compile(r"/?确认\s+([0-9a-fA-F]{8})")
ASK_PREFIX = re.compile(r"^(gemini|gpt|codex)[:：\s]+(.+)$", re.IGNORECASE | re.DOTALL)
HELP = ("可以发「策略状态」「最近回测」「Webull 模拟盘持仓」「富途持仓」，也可以直接和我聊天。\n"
        "问答模型：发「切换 Gemini」「切换 GPT」「切换自动」「当前模型」，或用「Gemini 问题」/「GPT 问题」单次指定。\n"
        "模拟盘限价单示例：买入 SOXL 2 股 限价 32.10。预览后回复确认码才会提交。\n"
        "回复「取消草稿」可清空待确认的订单。")


@dataclass
class Draft:
    code: str
    expires: float
    account_id: str
    intent: OrderIntent
    order: dict


class Chat:
    def __init__(self, config_path=None):
        self.sandbox = load_settings(config_path, environment="uat")
        self.client = None
        self.draft: Draft | None = None
        self.history = deque(maxlen=12)

    def _webull(self):
        if self.client is None and self.sandbox.has_webull_credentials:
            self.client = WebullClient(self.sandbox)
        return self.client

    def respond(self, text: str) -> str:
        text = text.strip()
        if text in {"帮助", "/help"}:
            return HELP
        if text in {"取消草稿", "/取消草稿"}:
            self.draft = None
            return "已清空待确认的订单草稿。"
        if match := CONFIRM.fullmatch(text):
            return self._confirm(match[1].lower())
        if match := ORDER.fullmatch(text):
            return self._preview(*match.groups())
        if text.startswith(("确认", "/确认", "下单", "/下单", "买入", "卖出")) or ("实盘" in text and any(k in text for k in ("买", "卖", "下单"))):
            return "聊天报单目前支持 Webull 模拟盘整股限价单。\n" + HELP
        if text in {"状态", "策略状态", "/状态"}:
            return chat_strategy.strategy_status(self.sandbox)
        if text in {"最近回测", "最新回测", "/回测"}:
            return chat_strategy.latest_backtest()
        if text in {"持仓", "Webull持仓", "Webull 模拟盘持仓", "Webull模拟盘持仓"}:
            client = self._webull()
            if client is None:
                return "Webull 模拟盘尚未配置凭证。"
            rows = client.positions()
            holdings = [f"{symbol} {row['quantity']:g} 股 · 成本 USD {row['cost_price']:.2f}"
                        for symbol, row in rows.items() if symbol in {"SOXL", "SOXS"}]
            return "Webull 模拟盘 SOXL／SOXS\n" + ("\n".join(holdings) or "暂无持仓")
        if text in {"富途持仓", "富途实盘持仓"}:
            return self._futu_positions()
        low = text.lower()
        if low in {"切换 gemini", "切换gemini", "/gemini", "用gemini", "用 gemini"}:
            if not gemini_key() and not gemini_chat.binary():
                return ("尚未配置 Gemini。可选其一：\n"
                        "1) 用个人 Google 账号登录官方 Gemini CLI：npm install -g @google/gemini-cli && gemini\n"
                        "2) 或配置 API Key：.venv/bin/python -m wquant wechat chat-gemini-setup")
            save_chat_provider("gemini")
            via = f"API（{gemini_chat.model_name()}）" if gemini_key() else "本机官方 Gemini CLI（个人账号）"
            return f"已切换为 Gemini {via} 回答私聊。"
        if low in {"切换 gpt", "切换gpt", "切换 codex", "切换codex", "/gpt", "/codex", "用gpt", "用codex"}:
            save_chat_provider("codex")
            return "已切换为 ChatGPT / Codex 回答私聊。"
        if low in {"切换自动", "切换 自动", "/auto"}:
            save_chat_provider("auto")
            return "已切换为自动模式（已配 Gemini Key 时优先 Gemini 秒回，失败自动回退 Codex/GPT）。"
        if text in {"当前模型", "模型状态", "/model"}:
            return self._model_status()
        if match := ASK_PREFIX.match(text):
            tag = match[1].lower()
            forced = "gemini" if tag == "gemini" else "codex"
            return self._answer(match[2].strip(), forced_provider=forced)
        return self._answer(text)

    def _preview(self, direction: str, symbol: str, quantity: str, price: str) -> str:
        client = self._webull()
        if client is None:
            return "Webull 模拟盘尚未配置凭证。"
        qty, limit = int(quantity), Decimal(price)
        if qty < 1 or limit <= 0:
            return "股数和限价都必须大于零。"
        self.draft = None
        intent = OrderIntent(symbol.upper(), "BUY" if direction == "买入" else "SELL", qty, float(limit), "微信手动下单")
        order = build_webull_order(intent, {"type": "LIMIT", "trading_session": "ALL_DAY", "limit_offset_bps": 0})
        try:
            account = client.account_id
            preview = client.preview_order(order)
            if isinstance(preview, list):
                preview = preview[0] if preview else None
            if not isinstance(preview, dict) or "estimated_cost" not in preview:
                raise RuntimeError("Webull 预检未返回预计订单成本")
        except Exception as exc:
            log.warning("微信订单预检失败（%s）", type(exc).__name__)
            return "券商预检未通过，请核对标的、股数、限价和该时段的交易权限。"
        code = token_hex(4)
        self.draft = Draft(code, monotonic() + 180, account, intent, order)
        return (f"Webull 模拟盘 · 尾号 {account[-4:]}\n{direction} {intent.symbol} {qty} 股 · 限价 USD {limit:.2f}\n"
                f"委托有效期 DAY · 时段 {order['support_trading_session']}\n"
                f"请确认是否提交。3 分钟内回复「确认 {code}」。")

    def _confirm(self, code: str) -> str:
        draft = self.draft
        if draft is None or draft.code != code:
            return "这个确认码没有对应的待确认订单。"
        self.draft = None
        if monotonic() >= draft.expires:
            return "订单预览已过期，请重新发起。"
        if self.client is None or self.client.account_id != draft.account_id:
            return "预览时的模拟盘账户已变化，请重新发起。"
        if session_for("ALL_DAY", datetime.now(ET)) != draft.order["support_trading_session"]:
            return "美股委托时段已切换，请重新预览确认。"
        order = draft.order
        client_id = order["client_order_id"]
        journal.record("wechat_manual_order", environment="uat", status="attempting", order=order)
        try:
            self.client.place_order(order)
        except Exception as exc:
            log.warning("微信模拟盘报单异常（%s）", type(exc).__name__)
            result = {"status": "error", "attempted": True, "order": order}
            reply = f"模拟盘报单状态待核对。单号 {client_id}。请先查券商订单，再决定是否重试。"
        else:
            result = {"status": "submitted", "order": order}
            reply = f"Webull 模拟盘已报单，成交状态待券商确认。单号 {client_id}"
        try:
            journal.record("wechat_manual_order", environment="uat", status=result["status"], order=order)
            wechat.record_order(self, draft.intent, result, include_initial=False)
        except Exception as exc:
            log.warning("微信模拟盘订单状态写入失败（%s）", type(exc).__name__)
        return reply

    @property
    def name(self):
        return "webull"

    @property
    def settings(self):
        return self.sandbox

    def _futu_positions(self):
        from . import futu_gateway as futu

        sdk = futu._futu()
        context = futu._trade_context()
        try:
            account = futu.select_account(futu.us_accounts(context), "REAL")
            ret, rows = context.position_list_query(position_market=sdk.TrdMarket.US, trd_env=sdk.TrdEnv.REAL,
                                                    acc_id=int(account["acc_id"]))
            if ret != sdk.RET_OK:
                return "富途暂时无法查询持仓。"
            positions = [f"{row['code']} {row['qty']:g} 股" for row in rows.to_dict("records")]
            return "富途美股实盘\n" + ("\n".join(positions) or "暂无持仓")
        finally:
            context.close()

    def _model_status(self) -> str:
        cur = chat_provider()
        gem_api = f"已配置（{gemini_chat.model_name()}）" if gemini_key() else "未配置"
        gem_cli = "可用（个人账号）" if gemini_chat.binary() else "未安装"
        cdx = "可用" if codex_chat.binary() else "不可用"
        oai = "已配置" if openai_key() else "未配置"
        return (f"当前默认问答引擎：{cur}\n"
                f"Gemini API：{gem_api}\n"
                f"本机 Gemini CLI：{gem_cli}\n"
                f"本机 Codex / ChatGPT：{cdx}\n"
                f"OpenAI API：{oai}")

    def _gemini_answer(self, messages, facts: str) -> str:
        key = gemini_key()
        if key:
            return gemini_chat.answer(messages, key=key, facts=facts)
        if gemini_chat.binary():
            return gemini_chat.cli_answer(messages, facts=facts)
        return ""

    def _codex_answer(self, messages, facts: str) -> str:
        return codex_chat.answer(messages, facts=facts)

    def _answer(self, text: str, *, forced_provider: str | None = None):
        user = {"role": "user", "content": text[:4000]}
        messages = [*self.history, user]
        facts = chat_strategy.facts_for(text)
        mode = forced_provider or chat_provider()
        if mode == "gemini":
            order = ("gemini", "codex", "openai") if forced_provider is None else ("gemini",)
        elif mode == "codex":
            order = ("codex", "openai", "gemini") if forced_provider is None else ("codex", "openai")
        elif mode == "openai":
            order = ("openai", "gemini", "codex")
        else:
            # auto: prefer fast Gemini API when configured, fallback to local Codex CLI -> OpenAI API
            order = ("gemini", "codex", "openai") if gemini_key() else ("codex", "openai", "gemini")

        answer = ""
        for engine in order:
            try:
                if engine == "gemini":
                    answer = self._gemini_answer(messages, facts)
                elif engine == "codex":
                    answer = self._codex_answer(messages, facts)
                elif engine == "openai":
                    answer = self._api_answer(messages)
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, requests.RequestException) as exc:
                log.warning("微信 %s 问答暂不可用（%s）", engine, type(exc).__name__)
                continue
            if answer:
                break

        if answer:
            self.history.extend((user, {"role": "assistant", "content": answer}))
            return answer
        if forced_provider == "gemini" and not gemini_key() and not gemini_chat.binary():
            return "尚未配置 Gemini（可安装官方 Gemini CLI 登录 Google 账号，或运行 wechat chat-gemini-setup 配置 API Key）。"
        return "AI 问答暂时没有回复。策略状态、回测和券商命令仍可使用。"

    def _api_answer(self, messages):
        key = openai_key()
        if not key:
            return ""
        response = requests.post("https://api.openai.com/v1/responses", headers={"Authorization": f"Bearer {key}"},
                                 json={"model": os.environ.get("OPENAI_CHAT_MODEL", "gpt-5-mini"), "store": False,
                                       "instructions": "你是私人微信对话助手。用中文自然、简洁地回答。你没有券商操作权限，也看不到实时账户。"
                                                       "不假装发出订单。交易预览、确认和查询由独立功能处理。",
                                       "input": messages, "max_output_tokens": 1200},
                                 timeout=35, allow_redirects=False)
        response.raise_for_status()
        return "".join(part["text"] for item in response.json().get("output", [])
                       for part in item.get("content", []) if part.get("type") == "output_text").strip()
