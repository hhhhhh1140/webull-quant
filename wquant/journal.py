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


def _owned_path(env: str) -> Path:
    return STATE / f"owned_{env}.json"


def owned(env: str) -> dict[str, int]:
    """系统自己买入的股数账本。账户里其他持仓（手动买的、共享沙盒里别人买的）系统一律不碰。"""
    p = _owned_path(env)
    return {k: int(v) for k, v in json.loads(p.read_text()).items()} if p.exists() else {}


def save_owned(env: str, d: dict[str, int]) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    _owned_path(env).write_text(json.dumps({k: v for k, v in d.items() if v > 0}, indent=2))


def entry_dates(key: str) -> dict[str, str]:
    """{symbol: 买入日期}，用于持有天数上限。key 区分账户（webull 环境 / paper）。"""
    p = STATE / f"entry_dates_{key}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def set_entry_date(key: str, symbol: str, day: str | None) -> None:
    d = entry_dates(key)
    if day is None:
        d.pop(symbol, None)
    else:
        d.setdefault(symbol, day)      # 加仓不重置日期
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / f"entry_dates_{key}.json").write_text(json.dumps(d, indent=2))


def hedges(key: str) -> dict[str, dict]:
    """当前的对冲仓位 {symbol: {entry_date, max_hold_days}}，退出规则和普通持仓不同。"""
    p = STATE / f"hedges_{key}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def set_hedge(key: str, symbol: str, info: dict | None) -> None:
    d = hedges(key)
    if info is None:
        d.pop(symbol, None)
    else:
        d[symbol] = info
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / f"hedges_{key}.json").write_text(json.dumps(d, indent=2))


def trail_highs() -> dict[str, float]:
    p = STATE / "trail_highs.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_trail_highs(d: dict[str, float]) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "trail_highs.json").write_text(json.dumps(d, indent=2))
