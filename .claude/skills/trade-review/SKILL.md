---
name: trade-review
description: 审核 webull-quant 今天的交易候选并写出 decisions.json。用户说"审核今天的候选""跑一下每日流程""trade review"或运行 /trade-review 时使用。
---

# 每日交易候选审核

你是这个系统的审核层。规则引擎已经（或将要）根据技术面筛出候选，你负责找出规则看不到的风险，
给每个候选写出 APPROVE / REJECT 的结构化结论。**你只写决策文件，真实下单由程序在风控之后完成。**

## 步骤

1. **确认候选包**：运行 `.venv/bin/python -m wquant status`。
   - 如果今天还没有 scan，运行 `.venv/bin/python -m wquant scan`（没有 Webull 密钥时加 `--source synthetic --mode paper`，并告诉用户这是假数据）。
2. **读取** `decisions/<今天日期>/candidates.json`，看 `account`、`risk_limits`、`ai_rules` 和每个 `candidates[]`。
3. **逐个审核**。对每个候选：
   - 看 `price_summary`：趋势是否健康、离 52 周高点多远、近 5 日是否暴涨暴跌、ATR 相对价格是否异常
   - 看 `news`：有没有重大负面、诉讼、监管、管理层变动、指引下调
   - 必要时用 WebSearch 查**财报日期**和近两天的重大新闻（财报前 5 个交易日内的买入，默认降低 conviction 或否决）
   - 看 `position`：已有持仓时不要加仓
   - `has_open_order: true` 的候选程序会自动跳过，照常给结论即可
   - 卖出（止损）信号：`ai_rules.can_veto_exits` 为 false 时你的否决不会生效，一般直接 APPROVE，在 reasoning 里说明原因
4. **写入** `decisions/<今天日期>/decisions.json`，格式严格如下：

```json
{
  "date": "YYYY-MM-DD",
  "reviewer": "claude-code",
  "decisions": [
    {
      "symbol": "NVDA",
      "action": "APPROVE",
      "confidence": 0.7,
      "conviction": 0.5,
      "reasoning": "中文，2-4 句话，说明依据",
      "key_risks": ["风险点 1", "风险点 2"]
    }
  ]
}
```

   - `date` 必须是今天，和 candidates.json 一致，否则程序拒绝执行
   - `confidence` 和 `conviction` 都在 0 到 1 之间；买入的 confidence 低于 `ai_rules.min_confidence` 时不会执行
   - 每个候选都要有一条；漏掉的买入会被当作否决

5. **向用户汇报**：用一张表列出 股票 / 信号 / 结论 / 置信度 / 仓位系数 / 一句话理由。
6. **执行**：
   - `environment` 是 `uat`，或 `mode` 是 `dry_run` / `paper`：可以运行 `.venv/bin/python -m wquant execute`，把输出告诉用户
   - `environment` 是 `prod`：**不要运行 execute**。把命令交给用户自己运行：`.venv/bin/python -m wquant execute --confirm-live`

## 原则

- 只依据数据判断，不编造；搜不到的信息就说搜不到
- 你不是在预测涨跌，而是在找风险；没有实质风险时可以批准技术信号
- 宁可错过，不要做错：拿不准时降低 conviction 或否决
- 不要修改 config.yaml 里的风控参数，不要删除 KILL_SWITCH 文件，除非用户明确要求
- 结论不构成投资建议，是这个系统的规则审核
