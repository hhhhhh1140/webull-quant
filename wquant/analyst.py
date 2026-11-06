"""审核层：决策的数据格式 + 可选的 Claude API 审核器。

默认（ai.mode: claude_code）由 Claude Code 在会话里审核候选并写 decisions.json，
程序不调用任何付费 API。只有 ai.mode: api 时才会用到下面的 ApiAnalyst。

原则：
- 审核只能在规则层的候选里批准 / 否决，不能凭空选股
- conviction 只能把仓位往下调，最终数量由风控层决定
"""
from __future__ import annotations

import json
import logging
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from .strategy import Signal

log = logging.getLogger(__name__)


class Decision(BaseModel):
    action: Literal["APPROVE", "REJECT"] = Field(description="是否执行规则引擎给出的信号")
    confidence: float = Field(ge=0, le=1)
    conviction: float = Field(ge=0, le=1, description="建议仓位比例，仅对买入有效")
    reasoning: str
    key_risks: list[str] = []


REVIEW_GUIDE = """你是一个美股日线级别交易系统里的风险审核员。

规则引擎已经根据技术面筛出了交易候选。你的工作是结合价格结构、近期新闻和持仓情况，
判断每个候选是否应该执行。你不是在预测股价，而是在找出规则引擎看不到的风险：
财报临近、重大负面新闻、监管事件、异常波动、新闻与走势背离等。

- 只依据提供的数据判断，不要编造数据里没有的事实
- 没有明显理由否决时，可以批准技术面信号；有实质性风险时要果断否决
- confidence 是你对这个判断本身的把握程度（0-1）
- conviction 是建议仓位比例（0-1，1 = 风控允许的最大仓位），风险越大越低
- reasoning 用中文，简洁，2-4 句话
"""


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


class ApiAnalyst:
    """仅在 ai.mode: api 时使用。每个候选调用一次 Claude API，会产生费用。"""

    def __init__(self, cfg: dict):
        import anthropic

        self.anthropic = anthropic
        self.cfg = cfg
        self.client = anthropic.Anthropic()

    def review(self, candidate: dict) -> Decision:
        sig = Signal(candidate["symbol"], candidate["action"], candidate["rule_reason"],
                     candidate["price"], candidate.get("atr", 0))
        user = ("请审核下面这个交易候选，返回结构化决策。\n\n"
                f"```json\n{json.dumps(candidate, ensure_ascii=False, indent=2, default=str)}\n```")
        a = self.anthropic
        try:
            resp = self.client.beta.messages.parse(
                model=self.cfg.get("api_model", "claude-opus-5-5"),
                max_tokens=16000,
                system=REVIEW_GUIDE,
                thinking={"type": "adaptive"},
                output_config={"effort": self.cfg.get("api_effort", "high")},
                output_format=Decision,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": user}],
            )
        except a.RateLimitError:
            return fallback_decision(sig, "Claude API 限流")
        except (a.APIConnectionError, a.APIStatusError) as e:
            log.warning("Claude API 出错：%s", e)
            return fallback_decision(sig, f"Claude API 出错: {type(e).__name__}")
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            return fallback_decision(sig, f"模型未给出有效决策 (stop_reason={resp.stop_reason})")
        return resp.parsed_output
