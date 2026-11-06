# webull-quant

基于 Webull 官方 OpenAPI 的美股/ETF 量化交易系统，**由 Claude Code 直接担任审核层**。程序本身不调用任何付费 API。

```
① scan     数据 (Webull 日线/新闻) → 规则层 strategy.py
           → decisions/<日期>/candidates.json        候选包
② 审核     Claude Code 运行 /trade-review，读候选包、必要时上网查财报和新闻
           → decisions/<日期>/decisions.json         批准/否决 + 置信度 + 仓位系数 + 风险点
③ execute  风控层 risk.py → 执行层 broker.py → logs/<日期>.jsonl
```

审核用的是你的 Claude 订阅额度（Pro/Max），不按 token 计费。
注意：如果你的终端环境里设置了 `ANTHROPIC_API_KEY`，Claude Code 会改用 API key 计费，审核前请先 `unset ANTHROPIC_API_KEY`。

审核层有三种模式（`config.yaml` → `ai.mode`）：

| 模式 | 谁来审核 | 费用 |
|---|---|---|
| `claude_code`（默认） | 你在 Claude Code 里运行 `/trade-review` | 订阅额度 |
| `api` | 程序每个候选调用一次 Claude API | 按 token 计费 |
| `off` | 不审核，规则信号直接进风控 | 无 |

## 设计原则

- **审核不能凭空选股**：只能在规则层给出的候选里批准或否决，并且只能把仓位往下调。
- **止损优先**：审核默认不能否决卖出信号（`ai.can_veto_exits: false`）。某个候选没有审核结论时，买入一律放弃，卖出照常执行。
- **决策过期作废**：`decisions.json` 的日期必须是今天，否则 execute 拒绝执行。
- **Claude 不碰实盘**：`WEBULL_ENVIRONMENT=prod` 时，Claude Code 只写决策，`execute --confirm-live` 必须由你自己运行。
- **最终下单数量由风控层决定**，回测和实盘用的是同一套风控代码。
- **实盘有三道锁**：`.env` 里 `WEBULL_ENVIRONMENT=prod` + `config.yaml` 里 `live_trading_enabled: true` + 命令行 `--confirm-live`。缺一个就拒绝下单。
- **紧急停止**：在项目根目录建一个名为 `KILL_SWITCH` 的空文件，系统就会停止开新仓。

## 安装

```bash
/opt/anaconda3/bin/python -m venv .venv      # 需要 Python 3.10+
.venv/bin/pip install -r requirements.txt
cp .env.example .env                          # 在本地编辑器里填写密钥，不要贴进聊天
```

`.env` 只需要填 `WEBULL_APP_KEY` / `WEBULL_APP_SECRET`（在 Webull 开发者后台申请）。

## 使用

```bash
# 1. 检查连接（默认连沙盒 uat；开了 2FA 的话，首次运行需要在手机 Webull App 上点批准）
.venv/bin/python -m wquant check

# 2. 拉取日线到 data/cache/
.venv/bin/python -m wquant fetch

# 3. 回测规则层
.venv/bin/python -m wquant backtest --trades 10

# 4. 每日流程（默认 dry_run，只打印不下单）
.venv/bin/python -m wquant scan       # 生成候选包
#    然后在 Claude Code 里输入 /trade-review —— Claude 审核并写 decisions.json
.venv/bin/python -m wquant execute    # 读审核结论 → 风控 → 下单
.venv/bin/python -m wquant status     # 随时查看今天的进度

# 5. 在沙盒里真正下单：把 config.yaml 的 mode 改成 webull（或命令加 --mode webull）
```

最省事的用法：直接在 Claude Code 里说"跑一下今天的流程"或输入 `/trade-review`，Claude 会自己 scan、审核、写决策；在沙盒/模拟模式下也会帮你执行。

不填密钥也能先跑通整个流程，用的是随机生成的假数据（结果没有任何参考意义）：

```bash
.venv/bin/python -m wquant backtest --source synthetic
.venv/bin/python -m wquant scan --source synthetic --mode paper
```

## 关于审核层能不能回测

**不能可靠地回测。** Claude 的训练数据里已经包含了这些历史行情和新闻，拿它去"回测" 2023 年的决策，等于让它看着答案考试，结果会严重高估。所以：

1. 回测只验证**规则层**（`wquant backtest`）
2. 审核层在**沙盒里前向测试**：每天跑一遍，积累几周到几个月的决策日志
3. 对比被否决的那些候选后来的走势，判断审核到底是在帮忙还是在添乱

`logs/*.jsonl` 里每一条都记录了规则理由、AI 的决定、置信度和风险点，方便做这种复盘。

## 定时运行

日线策略每天跑一次就够。建议美东时间 9:45（开盘 15 分钟后），用前一天的收盘数据下 DAY 限价单。

审核需要 Claude Code 参与，有两种方式：
- **手动**：每天打开 Claude Code，输入 `/trade-review`
- **自动**：在 Claude 桌面应用里建一个定时任务（工作日北京时间 22:45），内容是"在 ~/webull-quant 运行 /trade-review"。沙盒环境下它会一路执行到下单；实盘环境下它只写决策，最后一步由你自己运行

## 主要配置（config.yaml）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `risk.max_position_pct` | 0.10 | 单只股票最多占总资产的比例 |
| `risk.max_order_notional` | 5000 | 单笔订单金额上限，会和上一项一起生效，取较小值 |
| `risk.max_daily_loss_pct` | 0.03 | 当日亏损超过这个比例就停止开新仓 |
| `ai.mode` | claude_code | 审核方式，见上表 |
| `ai.min_confidence` | 0.6 | 批准买入时的最低置信度 |
| `order.type` | LIMIT | 限价单，价格 = 最新价 ± `limit_offset_bps` |

## 和官方 MCP / skills 怎么配合

这个系统直接调用官方 Python SDK（`webull-openapi-python-sdk`），下单的参数格式参照了官方 `webull-openapi-mcp`。

官方 MCP 更适合**人和 Claude 一起用**，比如查持仓、研究个股、复盘日志，可以另外配到 Claude Desktop 或 Claude Code 里。建议给 MCP 配上 `WEBULL_TOOLSETS=account,instrument,market-data`（不带交易工具）和 `WEBULL_SYMBOL_WHITELIST`，把自动下单只留给这个系统，责任清楚。

## 免责声明

本项目只是工程框架，示例策略不构成任何投资建议。上实盘之前，请先在沙盒里充分验证，并自己承担交易风险。
