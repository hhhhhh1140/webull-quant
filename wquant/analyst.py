"""Data models for archived individual-stock decisions."""
from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from .strategy import Signal

class Decision(BaseModel):
    action: Literal["APPROVE", "REJECT"] = Field(description="是否执行规则引擎给出的信号")
    confidence: float = Field(ge=0, le=1)
    conviction: float = Field(ge=0, le=1, description="建议仓位比例，仅对买入有效")
    reasoning: str
    key_risks: list[str] = []


class MarketView(BaseModel):
    regime: Literal["risk_on", "neutral", "risk_off"] = Field(description="偏多 / 中性 / 偏空")
    confidence: float = Field(ge=0, le=1)
    summary: str
    drivers: list[str] = []
    catalysts: list[str] = []
    # 事件前推演：[{event, consensus, scenarios: [{name, condition, expected_reaction, system_response}]}]
    event_scenarios: list[dict] = []
    # 事件后复盘：[{event, actual, market_reaction, matched_scenario, lesson}]
    event_reviews: list[dict] = []


class HedgeProposal(BaseModel):
    symbol: str
    conviction: float = Field(ge=0, le=1)
    reasoning: str
    key_risks: list[str] = []


def price_summary(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    close = df["close"]

    def r(x, n=2):
        return None if pd.isna(x) else round(float(x), n)

    return {
        "date": str(df.index[-1].date()),
        "close": r(last["close"]),
        "sma50": r(last["sma_fast"]),
        "sma200": r(last["sma_slow"]),
        "rsi14": r(last["rsi"], 1),
        "atr14": r(last["atr"]),
        "ret_5d": r(close.iloc[-1] / close.iloc[-6] - 1, 4) if len(close) > 6 else None,
        "ret_20d": r(last["momentum"], 4),
        "ret_60d": r(close.iloc[-1] / close.iloc[-61] - 1, 4) if len(close) > 61 else None,
        "high_52w": r(close.tail(252).max()),
        "low_52w": r(close.tail(252).min()),
        "last_10_closes": [r(x) for x in close.tail(10)],
    }


def fallback_decision(signal: Signal, why: str) -> Decision:
    """没有有效审核结论时：买入一律放弃（宁可错过）；卖出照常执行（止损优先）。"""
    if signal.action == "SELL":
        return Decision(action="APPROVE", confidence=1.0, conviction=1.0, reasoning=f"{why}；卖出信号默认执行")
    return Decision(action="REJECT", confidence=0.0, conviction=0.0, reasoning=f"{why}；买入信号默认放弃", key_risks=[why])
