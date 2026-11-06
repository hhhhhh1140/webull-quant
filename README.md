# webull-quant

基于 Webull 官方 OpenAPI 的美股/ETF 量化交易系统，加入 Claude 做 AI 辅助决策。

```
数据 (Webull 日线/新闻)
  → 规则层 strategy.py      趋势+动量筛选，产生 BUY 候选 / SELL 止损信号
  → AI 层   analyst.py      Claude 审核候选：批准/否决 + 仓位系数 + 风险点（结构化输出）
  → 风控层  risk.py         白名单、单仓上限、单笔金额上限、最大持仓数、当日亏损熔断、KILL_SWITCH
  → 执行层  broker.py       dry_run / paper（本地模拟）/ webull（先 preview 再 place）
  → 日志    journal.py      logs/YYYY-MM-DD.jsonl 记录每一个决策
```

## 设计原则

- **AI 不能凭空选股**：只能在规则层给出的候选里批准或否决，并且只能把仓位往下调。
- **止损优先**：AI 默认不能否决卖出信号（`ai.can_veto_exits: false`）。AI 出错或被限流时，买入一律放弃，卖出照常执行。
- **最终下单数量由风控层决定**，回测和实盘用的是同一套风控代码。
- **实盘有三道锁**：`.env` 里 `WEBULL_ENVIRONMENT=prod` + `config.yaml` 里 `live_trading_enabled: true` + 命令行 `--confirm-live`。缺一个就拒绝下单。
- **紧急停止**：在项目根目录建一个名为 `KILL_SWITCH` 的空文件，系统就会停止开新仓。

## 安装

```bash
/opt/anaconda3/bin/python -m venv .venv      # 需要 Python 3.10+
.venv/bin/pip install -r requirements.txt
cp .env.example .env                          # 在本地编辑器里填写密钥，不要贴进聊天
```

`.env` 需要填：
- `WEBULL_APP_KEY` / `WEBULL_APP_SECRET`：在 Webull 开发者后台申请
- `ANTHROPIC_API_KEY`：AI 层要用；也可以运行 `ant auth login` 登录代替

## 使用

```bash
# 1. 检查连接（默认连沙盒 uat；开了 2FA 的话，首次运行需要在手机 Webull App 上点批准）
.venv/bin/python -m wquant check

# 2. 拉取日线到 data/cache/
.venv/bin/python -m wquant fetch

# 3. 回测规则层
.venv/bin/python -m wquant backtest --trades 10

# 4. 执行一次每日流程（默认 dry_run，只打印不下单）
.venv/bin/python -m wquant run

# 5. 在沙盒里真正下单：把 config.yaml 的 mode 改成 webull，或者
.venv/bin/python -m wquant run --mode webull
```

不填密钥也能先跑通整个流程，用的是随机生成的假数据（结果没有任何参考意义）：

```bash
.venv/bin/python -m wquant backtest --source synthetic
.venv/bin/python -m wquant run --source synthetic --mode paper --no-ai
```

## 关于 AI 层能不能回测

**不能可靠地回测。** Claude 的训练数据里已经包含了这些历史行情和新闻，拿它去"回测" 2023 年的决策，等于让它看着答案考试，结果会严重高估。所以：

1. 回测只验证**规则层**（`wq backtest`）
2. AI 层在**沙盒里前向测试**：每天跑 `run --mode webull`（uat 环境），积累几周到几个月的决策日志
3. 对比 AI 否决的那些候选后来的走势，判断 AI 到底是在帮忙还是在添乱

`logs/*.jsonl` 里每一条都记录了规则理由、AI 的决定、置信度和风险点，方便做这种复盘。

## 定时运行

日线策略每天跑一次就够。建议美东时间 9:45（开盘 15 分钟后），用前一天的收盘数据下 DAY 限价单。macOS 用 cron 的例子：

```cron
# 北京时间 22:45（美东夏令时 9:45），周一到周五
45 22 * * 1-5 cd ~/webull-quant && .venv/bin/python -m wquant run >> logs/cron.log 2>&1
```

## 主要配置（config.yaml）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `risk.max_position_pct` | 0.10 | 单只股票最多占总资产的比例 |
| `risk.max_order_notional` | 5000 | 单笔订单金额上限，会和上一项一起生效，取较小值 |
| `risk.max_daily_loss_pct` | 0.03 | 当日亏损超过这个比例就停止开新仓 |
| `ai.min_confidence` | 0.6 | AI 批准买入时的最低置信度 |
| `ai.model` | claude-opus-5-5 | 每次 AI 审核都会调用一次 Claude API，会产生费用 |
| `order.type` | LIMIT | 限价单，价格 = 最新价 ± `limit_offset_bps` |

## 和官方 MCP / skills 怎么配合

这个系统直接调用官方 Python SDK（`webull-openapi-python-sdk`），下单的参数格式参照了官方 `webull-openapi-mcp`。

官方 MCP 更适合**人和 Claude 一起用**，比如查持仓、研究个股、复盘日志，可以另外配到 Claude Desktop 或 Claude Code 里。建议给 MCP 配上 `WEBULL_TOOLSETS=account,instrument,market-data`（不带交易工具）和 `WEBULL_SYMBOL_WHITELIST`，把自动下单只留给这个系统，责任清楚。

## 免责声明

本项目只是工程框架，示例策略不构成任何投资建议。上实盘之前，请先在沙盒里充分验证，并自己承担交易风险。
