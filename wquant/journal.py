"""交易日志：每一次信号、AI 决策、风控结论、下单结果都写进 logs/YYYY-MM-DD.jsonl。"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from .config import ROOT

LOGS = ROOT / "logs"
STATE = ROOT / "state"


def record(event: str, **data) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    line = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event, **data}
    with open(LOGS / f"{date.today()}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")


def start_of_day_equity(current: float) -> float:
    """当天第一次运行时记录权益，用于当日亏损熔断。"""
    STATE.mkdir(parents=True, exist_ok=True)
    p = STATE / f"sod_equity_{date.today()}.json"
    if p.exists():
        return float(json.loads(p.read_text())["equity"])
    p.write_text(json.dumps({"equity": current}))
    return current


def trail_highs() -> dict[str, float]:
    p = STATE / "trail_highs.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_trail_highs(d: dict[str, float]) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "trail_highs.json").write_text(json.dumps(d, indent=2))
