from types import SimpleNamespace

from wquant import chat_trade, clawbot, codex_chat


def test_webull_order_requires_exact_confirmation_once(monkeypatch):
    chat = chat_trade.Chat.__new__(chat_trade.Chat)
    chat.draft = None
    chat.sandbox = SimpleNamespace(environment="uat")
    previews, orders = [], []
    def preview(order):
        previews.append(order)
        return [{"estimated_cost": "64.20"}]

    chat.client = SimpleNamespace(account_id="00001234", preview_order=preview, place_order=orders.append)
    monkeypatch.setattr(chat_trade.journal, "record", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_trade.wechat, "record_order", lambda *args, **kwargs: None)
    assert "确认" in chat.respond("买入 SOXL 2 股 限价 32.10")
    assert len(previews) == 1 and orders == []
    code = chat.draft.code
    assert "尾号 1234" in chat.respond("买入 SOXL 2 股 限价 32.10")
    assert "没有对应" in chat.respond(f"确认 {code}")
    assert orders == []
    code = chat.draft.code
    assert "已报单" in chat.respond(f"确认 {code}")
    assert "没有对应" in chat.respond(f"确认 {code}")
    assert len(orders) == 1 and orders[0] == previews[-1]
    assert orders[0]["limit_price"] == "32.10"
    assert "模拟盘" in chat.respond("实盘买入 SOXL 2 股 限价 32.10")
    assert len(previews) == 2 and len(orders) == 1


def test_inbound_accepts_paired_owner_only_and_deduplicates(monkeypatch, tmp_path):
    monkeypatch.setattr(clawbot, "STATE", tmp_path)
    account = {"owner": "owner", "token": "token"}
    sent, received = [], []
    monkeypatch.setattr(clawbot, "_send", lambda account, context, answer: sent.append((context, answer)))

    def respond(text):
        received.append(text)
        return "收到"

    chat = SimpleNamespace(respond=respond)
    message = {"from_user_id": "owner", "message_type": 1, "message_id": "1", "context_token": "context",
               "item_list": [{"type": 1, "text_item": {"text": "帮助"}}]}
    clawbot.handle(account, {}, [{**message, "from_user_id": "other"},
                                {**message, "group_id": "room"}, {**message, "message_type": 2}, message], chat)
    clawbot.handle(account, clawbot._read(clawbot._sync_path(account)), [message], chat)
    assert received == ["帮助"]
    assert sent == [("context", "收到")]
    assert clawbot._sync_path(account).stat().st_mode & 0o777 == 0o600


def test_codex_chat_does_not_inherit_broker_credentials(monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("WEBULL_APP_SECRET", "do-not-forward")
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-forward")
    monkeypatch.setattr(codex_chat, "binary", lambda: "/test/codex")

    def run(argv, **options):
        assert argv[0] == "/test/codex" and "--ephemeral" in argv
        assert argv[argv.index("--sandbox") + 1] == "read-only"
        assert "shell_tool" in argv and "unified_exec" in argv and "mcp_servers={}" in argv
        assert "WEBULL_APP_SECRET" not in options["env"] and "OPENAI_API_KEY" not in options["env"]
        assert "回测摘要" in options["input"]
        Path(argv[argv.index("--output-last-message") + 1]).write_text("已答复")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(codex_chat.subprocess, "run", run)
    assert codex_chat.answer([{"role": "user", "content": "最近如何"}], facts="回测摘要") == "已答复"


def test_gemini_chat_request_and_wechat_provider_switching(monkeypatch, tmp_path):
    from collections import deque
    from wquant import gemini_chat

    monkeypatch.setattr(clawbot, "STATE", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("WQUANT_CHAT_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "Gemini 正常回复"}]}}]}

    def fake_post(url, *, headers, json, timeout, allow_redirects):
        captured.update(url=url, headers=headers, json=json)
        return FakeResp()

    monkeypatch.setattr(gemini_chat.requests, "post", fake_post)
    monkeypatch.setattr(codex_chat, "answer", lambda msgs, facts="": "Codex 回复")
    monkeypatch.setattr(codex_chat, "binary", lambda: "/test/codex")

    chat = chat_trade.Chat.__new__(chat_trade.Chat)
    chat.draft = None
    chat.sandbox = SimpleNamespace(environment="uat")
    chat.client = None
    chat.history = deque(maxlen=12)

    # Without key, switching to Gemini prompts setup
    monkeypatch.setattr(gemini_chat, "binary", lambda: None)
    assert "chat-gemini-setup" in chat.respond("切换 Gemini")

    # Official Gemini CLI (personal Google account) works without API key and never forwards broker secrets
    monkeypatch.setenv("WEBULL_APP_SECRET", "secret-must-not-leak")
    monkeypatch.setattr(gemini_chat, "binary", lambda: "/test/bin/gemini")
    def fake_gemini_cli(argv, **opts):
        assert argv[0] == "/test/bin/gemini" and argv[1] == "-p"
        assert "WEBULL_APP_SECRET" not in opts["env"]
        assert "/test/bin" in opts["env"]["PATH"]
        return SimpleNamespace(returncode=0, stdout="Gemini CLI 个人账号回复\n", stderr="")
    monkeypatch.setattr(gemini_chat.subprocess, "run", fake_gemini_cli)
    assert "本机官方 Gemini CLI" in chat.respond("切换 Gemini")
    assert chat.respond("你好") == "Gemini CLI 个人账号回复"

    # Save Gemini key -> switch works and replies via Gemini
    clawbot.save_gemini_key("test-gem-key")
    assert "已切换为 Gemini" in chat.respond("切换 Gemini")
    assert clawbot.chat_provider() == "gemini"
    assert chat.respond("SOXL 今天怎么看") == "Gemini 正常回复"
    assert captured["headers"]["x-goog-api-key"] == "test-gem-key"
    assert "gemini-2.5-flash:generateContent" in captured["url"]

    # One-off prefix overrides provider without changing saved preference
    assert chat.respond("GPT 帮我总结一下") == "Codex 回复"
    assert clawbot.chat_provider() == "gemini"

    # Switch to Codex and test one-off Gemini prefix
    assert "ChatGPT / Codex" in chat.respond("切换 GPT")
    assert chat.respond("你好") == "Codex 回复"
    assert chat.respond("Gemini 你好") == "Gemini 正常回复"

    # Fallback: if Gemini errors in auto mode, falls back to Codex cleanly
    clawbot.save_chat_provider("auto")
    monkeypatch.setattr(gemini_chat, "answer", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("quota")))
    assert chat.respond("测试自动回退") == "Codex 回复"
    assert "Gemini API：已配置" in chat.respond("当前模型")
