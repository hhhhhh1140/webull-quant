"""Use the Mac's existing Codex login for private WeChat text replies."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

APP_CLI = Path("/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex")


def binary() -> str | None:
    return os.environ.get("WQUANT_CODEX_BIN") or shutil.which("codex") or (str(APP_CLI) if APP_CLI.is_file() else None)


def answer(messages, *, facts="") -> str:
    executable = binary()
    if executable is None:
        raise FileNotFoundError("找不到 Codex CLI")
    prompt = ("你是私人微信问答助手。用中文自然、简洁地回答。只使用以下文字，"
              "无需调用工具。券商查询、交易和确认由微信的独立命令处理。"
              "没有实时行情或账户证据时直接说明，不宣称修改文件或执行交易。\n"
              f"明确提供的策略资料：{facts or '无'}\n"
              "按顺序排列的私聊问答，最后一条是当前问题：\n"
              + json.dumps(messages, ensure_ascii=False))
    env = {key: os.environ[key] for key in ("HOME", "USER", "LOGNAME", "PATH", "TMPDIR", "LANG", "LC_ALL", "CODEX_HOME")
           if key in os.environ}
    with TemporaryDirectory(prefix="wquant-codex-qa-") as directory:
        output = Path(directory) / "answer.txt"
        result = subprocess.run(
            [executable, "exec", "--cd", directory, "--skip-git-repo-check", "--ephemeral",
             "--ignore-user-config", "--sandbox", "read-only", "--disable", "shell_tool",
             "--disable", "unified_exec", "--disable", "multi_agent", "--disable", "multi_agent_v2",
             "--disable", "image_generation", "--disable", "view_image", "--enable", "skip_host_skill_discovery",
             "-c", 'approval_policy="never"', "-c", "mcp_servers={}", "-c", 'web_search="disabled"',
             "--color", "never", "--output-last-message", str(output), "-"],
            input=prompt, text=True, env=env, cwd=directory, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=70, check=False,
        )
        if result.returncode != 0 or not output.is_file():
            raise RuntimeError("Codex 未能返回回复")
        return output.read_text(encoding="utf-8").strip()
