"""Local strategy summaries safe to show in the WeChat conversation."""
from __future__ import annotations

import json

from .config import ROOT


def strategy_status(settings) -> str:
    from .paper_arena import arena_snapshot

    lines = ["三套策略的本地模拟账本"]
    for row in arena_snapshot(settings):
        gate = "开" if row["macro_gate_enabled"] else "关"
        lines.append(f"{row['label']}｜账面权益 USD {row['equity']:,.2f}（{row['return_pct']:+.2%}）\n"
                     f"{row['holding']}｜配置的利率门控 {gate}")
    return "\n".join(lines)


def latest_backtest() -> str:
    reports = (ROOT / "reports").glob("dual-layer-*/summary.json")
    for report in sorted(reports, key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            summary = json.loads(report.read_text(encoding="utf-8"))
            full, recent = summary["2020_2026_hybrid"], summary["2026_ytd_hybrid"]
            stats = full["stats"]
            return (f"双层回测报告 {report.parent.name}\n"
                    f"{full['start']} 至 {full['end']}：收益 {full['total']:+.2%}，最大回撤 {abs(full['max_drawdown']):.2%}\n"
                    f"2026 年同期：收益 {recent['total']:+.2%}，最大回撤 {abs(recent['max_drawdown']):.2%}\n"
                    f"长样本使用 1 分钟 {stats.get('days_M1', 0)} 天、5 分钟 {stats.get('days_M5', 0)} 天、"
                    f"日线 {stats.get('days_D', 0)} 天。")
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return "本地暂无可读取的双层回测。"


def facts_for(question: str) -> str:
    if not any(word in question.upper() for word in ("策略", "回测", "富途", "SOXL", "SOXS", "A版", "B版")):
        return ""
    from .paper_arena import PROFILES

    rules = "\n".join(f"{profile.label}：{profile.summary}" for profile in PROFILES)
    return f"{rules}\n{latest_backtest()}\n报告仅代表该回测口径，无法据此给新版 A/B 单独归因。"
