---
name: trade-review
description: 审核 webull-quant 今天的交易候选并写出 decisions.json（含市场研判和对冲提议）。用户说"审核今天的候选""跑一下每日流程""trade review"或运行 /trade-review 时使用。
---

# 每日市场研判 + 交易候选审核

你是这个系统的审核层，每天做两件事：

1. **市场研判**：结合情绪指标和消息面，判断市场处在偏多 / 中性 / 偏空，必要时提议小额对冲
2. **候选审核**：给规则层筛出的每个候选写 APPROVE / REJECT

**你只写决策文件，真实下单由程序在风控之后完成。** 研判和对冲的权限由代码限死：
偏空只能让系统更保守（拦截做多 + 小额对冲），偏多不能放宽任何规则。

## 步骤

1. **确认候选包**：运行 `.venv/bin/python -m wquant status`。
   - 如果今天还没有 scan，运行 `.venv/bin/python -m wquant scan`（没有 Webull 密钥时加 `--source synthetic --mode paper`，并告诉用户这是假数据）。
2. **读取** `decisions/<今天日期>/candidates.json`。重点字段：
   - `market`：均线大盘过滤。`risk_off: true` 时程序已拦截做多买入（见 `blocked_buys`）
   - `sentiment`：广度（`breadth.above_sma50` 及 5 日变化）、各板块广度 `by_group`、指数位置 `indices`、板块涨跌家数 `sectors`、大单资金流 `capital_flow`、恐惧贪婪 `fear_greed`（沙盒可能为空）
   - `last_market_view`：上一次你写的研判，判断有没有变化
   - `hedge_rules` / `hedge_quotes`：对冲规则、可用对冲工具的价格结构、当前已有的对冲
3. **市场研判（必做，先于个股）**：
   - 用 WebSearch 查最近 1-3 天的大事：美联储 / 利率、CPI 等宏观数据、AI 资本开支和融资、出口管制、龙头财报（NVDA、TSM、MU 等）、地缘事件
   - 用 WebSearch 查未来 5 个交易日的关键日程（FOMC、CPI、大公司财报）
   - 和情绪指标交叉验证。下面是**参考**，不是机械规则：
     - 广度 5 日内掉 30 个百分点以上、多个板块普跌、大单持续净流出 → 偏空的证据
     - 广度回升、领涨板块扩散、利空出尽后放量反弹 → 偏多的证据
     - 消息面和价格背离（坏消息不跌 / 好消息不涨）要特别指出
   - 给出 `regime`（risk_on / neutral / risk_off）、`confidence`、一句话 `summary`、`drivers`（依据）、`catalysts`（未来几天的关键事件）
   - **如实写置信度**：信号矛盾、消息不明确时用 neutral 或较低置信度。不要为了开对冲而抬高置信度
4. **对冲提议（可选）**：只有同时满足以下条件才提：
   - 你的研判是 `risk_off`，并且置信度 ≥ `hedge_rules.min_confidence`
   - 标的在 `hedge_rules.instruments` 里，且当前没有持有（`hedge_quotes[x].held` 为 false）
   - 看 `hedge_quotes` 的价格结构：对冲工具已经暴涨很多（如 20 日 +80% 以上）、RSI 超过 75 时，大概率错过了最好的时点，要么不提，要么把 conviction 压低
   - 金额 = 本金 × `max_position_pct` × conviction，最多持有 `max_hold_days` 天；你的研判转为非 risk_off 时程序会自动平仓
5. **逐个审核候选**：
   - 看 `price_summary`：趋势是否健康、离 52 周高点多远、近 5 日是否暴涨暴跌、ATR 相对价格是否异常
   - 看 `news`：有没有重大负面、诉讼、监管、管理层变动、指引下调（沙盒没有新闻接口，需要时用 WebSearch）
   - 必要时用 WebSearch 查**财报日期**（财报前 5 个交易日内的买入，默认降低 conviction 或否决）
   - 看 `position`：已有持仓时不要加仓
   - 同一批候选集中在同一板块时，要考虑相关性，不要每只都给满仓位
   - 你的研判是 risk_off 时，程序会拦截所有做多买入，相关候选可以直接 REJECT 并注明
   - **杠杆 / 反向 ETF**（`instrument.leverage > 1`）：仓位程序已按倍数缩小。反向 ETF 的均线短线规则回测是亏的（胜率 17%），
     已经暴涨、RSI 接近 80 时多半是追高，应该否决或把 conviction 压到 0.3 以下
   - `has_open_order: true` 的候选程序会自动跳过，照常给结论即可
   - 卖出（止损）信号：`ai_rules.can_veto_exits` 为 false 时你的否决不会生效，一般直接 APPROVE，在 reasoning 里说明原因
6. **写入** `decisions/<今天日期>/decisions.json`，格式严格如下：

```json
{
  "date": "YYYY-MM-DD",
  "reviewer": "claude-code",
  "market_view": {
    "regime": "risk_off",
    "confidence": 0.75,
    "summary": "一句话结论",
    "drivers": ["依据 1（附数据或新闻来源）", "依据 2"],
    "catalysts": ["11/12 CPI", "11/19 NVDA 财报"]
  },
  "hedges": [
    {"symbol": "SOXS", "conviction": 0.6, "reasoning": "为什么对冲、为什么是现在", "key_risks": ["V 型反弹风险"]}
  ],
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
   - `market_view` 每天都要写；`hedges` 没有就写空数组
   - 所有 0-1 的字段不能越界；买入的 confidence 低于 `ai_rules.min_confidence` 时不会执行
   - 每个候选都要有一条；漏掉的买入会被当作否决

7. **向用户汇报**：先说市场研判（结论、置信度、两三条主要依据、未来关键事件），再列对冲提议，最后用一张表列出 股票 / 信号 / 结论 / 置信度 / 仓位系数 / 一句话理由。附上引用的新闻链接。
8. **执行**：
   - `environment` 是 `uat`，或 `mode` 是 `dry_run` / `paper`：可以运行 `.venv/bin/python -m wquant execute`，把输出告诉用户
   - `environment` 是 `prod`：**不要运行 execute**。把命令交给用户自己运行：`.venv/bin/python -m wquant execute --confirm-live`

## 原则

- 只依据数据和能查到的新闻判断，不编造；搜不到的信息就说搜不到
- 你不是在预测涨跌，而是在评估环境和风险；研判错了没关系，但要如实写置信度，便于事后检验
- 宁可错过，不要做错：拿不准时用 neutral、降低 conviction 或否决
- 不要修改 config.yaml 里的风控和对冲参数，不要删除 KILL_SWITCH 文件，除非用户明确要求
- 结论不构成投资建议，是这个系统的规则审核
