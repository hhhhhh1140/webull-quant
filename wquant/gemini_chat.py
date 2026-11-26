"""Gemini replies for private WeChat chat via Gemini API or official Google Gemini CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import requests

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-2.5-flash"


def model_name() -> str:
    return (os.environ.get("GEMINI_CHAT_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def binary() -> str | None:
    """Locate the official Google Gemini CLI binary, including Homebrew and nvm paths for launchd."""
    env_bin = (os.environ.get("WQUANT_GEMINI_BIN") or "").strip()
    if env_bin and Path(env_bin).is_file():
        return env_bin
    found = shutil.which("gemini")
    if found:
        return found
    candidates = [
        Path("/opt/homebrew/bin/gemini"),
        Path("/usr/local/bin/gemini"),
        Path.home() / ".npm-global/bin/gemini",
    ]
    nvm_dir = Path.home() / ".nvm/versions/node"
    if nvm_dir.is_dir():
        candidates.extend(sorted(nvm_dir.glob("*/bin/gemini"), reverse=True))
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def cli_answer(messages: list[dict], *, facts: str = "") -> str:
    """Use the local official Google Gemini CLI (logged in with personal Google account)."""
    executable = binary()
    if executable is None:
        raise FileNotFoundError("找不到官方 Gemini CLI")
    prompt = (
        "你是私人微信问答助手。用中文自然、简洁地回答。只使用以下文字，"
        "不要调用任何本地工具、不要读写文件或执行命令。券商查询、交易和确认由微信独立命令处理。"
        "没有实时行情或账户证据时直接说明，不宣称修改文件或执行交易。\n"
        f"明确提供的策略资料：{facts or '无'}\n"
        "按顺序排列的私聊问答，最后一条是当前问题：\n"
        + json.dumps(messages, ensure_ascii=False)
    )
    env = {
        key: os.environ[key]
        for key in ("HOME", "USER", "LOGNAME", "PATH", "TMPDIR", "LANG", "LC_ALL",
                    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
        if key in os.environ
    }
    # Ensure nvm/homebrew node next to the gemini binary is reachable under launchd
    bin_dir = str(Path(executable).parent)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '/usr/bin:/bin')}"
    with TemporaryDirectory(prefix="wquant-gemini-qa-") as directory:
        result = subprocess.run(
            [executable, "-p", prompt],
            text=True,
            env=env,
            cwd=directory,
            capture_output=True,
            timeout=70,
            check=False,
        )
        out = (result.stdout or "").strip()
        if result.returncode != 0 or not out:
            raise RuntimeError("Gemini CLI 未能返回回复")
        return out


def verify_api_key(key: str) -> None:
    response = requests.get(
        f"{API_BASE}/models",
        params={"pageSize": 1},
        headers={"x-goog-api-key": key},
        timeout=10,
        allow_redirects=False,
    )
    response.raise_for_status()


def answer(messages: list[dict], *, key: str, facts: str = "", model: str | None = None) -> str:
    if not key:
        return ""
    chosen = (model or model_name()).strip()
    instruction = (
        "你是私人微信对话助手。用中文自然、简洁地回答。"
        "你没有券商操作权限，也看不到实时账户；交易预览、确认和查询由微信独立命令处理。"
        "没有实时行情或账户证据时直接说明，不宣称修改文件或执行交易。"
    )
    if facts:
        instruction += f"\n明确提供的策略资料：\n{facts}"

    contents = [
        {
            "role": "model" if msg.get("role") == "assistant" else "user",
            "parts": [{"text": str(msg.get("content", ""))[:4000]}],
        }
        for msg in messages
        if msg.get("content")
    ]
    if not contents:
        return ""

    response = requests.post(
        f"{API_BASE}/models/{chosen}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={
            "systemInstruction": {"parts": [{"text": instruction}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 1200},
        },
        timeout=35,
        allow_redirects=False,
    )
    response.raise_for_status()
    data = response.json()
    texts = []
    for cand in data.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            if isinstance(part, dict) and part.get("text"):
                texts.append(part["text"])
    return "".join(texts).strip()
