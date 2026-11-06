"""AI 分析层：让 Claude 审核规则层给出的候选，输出结构化决策。

原则：
- AI 只能在规则层的候选里做取舍（批准 / 否决 / 调低仓位），不能凭空选股
- 输出是有类型约束的结构化 JSON（Pydantic），不解析自由文本
- 最终下单数量永远由风控层决定，AI 给的 conviction 只能把仓位往下调
"""
from __future__ import annotations

import json
import logging
from typing import Literal

import anthropic
import pandas as pd
from pydantic import BaseModel, Field

from .strategy import Signal

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个美股日线级别交易系统里的风险审核分析师。

规则引擎已经根据技术面筛出了一个交易候选。你的工作是结合价格结构、近期新闻和持仓情况，
判断这笔交易是否应该执行。你不是在预测股价，而是在找出规则引擎看不到的风险：
财报临近、重大负面新闻、监管事件、异常波动、新闻与走势背离等。

要求：
- 只依据下面提供的数据判断，不要编造数据里没有的事实
- 没有明显理由否决时，可以批准技术面信号；有实质性风险时要果断否决
- confidence 是你对这个判断本身的把握程度（0-1）
- conviction 是建议仓位比例（0-1，1 = 风控允许的最大仓位）。风险越大越低
- reasoning 用中文，简洁，2-4 句话
"""


class Decision(BaseModel):
    action: Literal["APPROVE", "REJECT"] = Field(description="是否执行规则引擎给出的信号")
    confidence: float = Field(ge=0, le=1)
    conviction: float = Field(ge=0, le=1, description="建议仓位比例，仅对买入有效")
    reasoning: str
    key_risks: list[str]


def _price_summary(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    close = df["close"]
    return {
        "date": str(df.index[-1].date()),
        "close": round(float(last["close"]), 2),
        "sma50": round(float(last["sma_fast"]), 2),
        "sma200": round(float(last["sma_slow"]), 2),
        "rsi14": round(float(last["rsi"]), 1),
        "atr14": round(float(last["atr"]), 2),
        "ret_5d": round(float(close.iloc[-1] / close.iloc[-6] - 1), 4) if len(close) > 6 else None,
        "ret_20d": round(float(last["momentum"]), 4),
        "ret_60d": round(float(close.iloc[-1] / close.iloc[-61] - 1), 4) if len(close) > 61 else None,
        "high_52w": round(float(close.tail(252).max()), 2),
        "low_52w": round(float(close.tail(252).min()), 2),
        "last_10_closes": [round(float(x), 2) for x in close.tail(10)],
    }


class Analyst:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client = anthropic.Anthropic()

    def review(self, signal: Signal, df: pd.DataFrame, news: list[dict], position: dict | None) -> Decision:
        payload = {
            "signal": {"symbol": signal.symbol, "action": signal.action, "rule_reason": signal.reason},
            "price": _price_summary(df),
            "position": position,
            "recent_news": news,
        }
        user = (
            "请审核下面这个交易候选，返回结构化决策。\n\n"
            f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}\n```"
        )
        try:
            resp = self.client.beta.messages.parse(
                model=self.cfg.get("model", "claude-opus-5-5"),
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                output_config={"effort": self.cfg.get("effort", "high")},
                output_format=Decision,
                # 模型因安全策略拒答时，服务端自动换模型重试
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.RateLimitError:
            log.warning("Claude API 限流，%s 按保守处理（否决）", signal.symbol)
            return _fallback_decision(signal, "Claude API 限流")
        except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:
            log.warning("Claude API 出错：%s", e)
            return _fallback_decision(signal, f"Claude API 出错: {type(e).__name__}")

        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            return _fallback_decision(signal, f"模型未给出有效决策 (stop_reason={resp.stop_reason})")
        return resp.parsed_output


def _fallback_decision(signal: Signal, why: str) -> Decision:
    """AI 不可用时：买入一律否决（宁可错过）；卖出照常执行（止损优先）。"""
    if signal.action == "SELL":
        return Decision(action="APPROVE", confidence=1.0, conviction=1.0, reasoning=f"{why}；卖出信号默认执行", key_risks=[])
    return Decision(action="REJECT", confidence=0.0, conviction=0.0, reasoning=f"{why}；买入信号默认放弃", key_risks=[why])
