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
.venv/bin/python -m wquant orders     # 查看本系统在 Webull 下的订单

# 5. 在沙盒里真正下单：把 config.yaml 的 mode 改成 webull（或命令加 --mode webull）
```

最省事的用法：直接在 Claude Code 里说"跑一下今天的流程"或输入 `/trade-review`，Claude 会自己 scan、审核、写决策；在沙盒/模拟模式下也会帮你执行。

不填密钥也能先跑通整个流程，用的是随机生成的假数据（结果没有任何参考意义）：

```bash
.venv/bin/python -m wquant backtest --source synthetic
.venv/bin/python -m wquant scan --source synthetic --mode paper
```

## 市场研判与对冲

每天审核时，Claude 先写一份**市场研判**（偏多 / 中性 / 偏空 + 置信度 + 依据 + 未来关键事件），再审核个股。依据包括：

- 程序计算的情绪指标（写在候选包的 `sentiment` 里）：市场广度及其 5 日变化、各板块广度、指数位置、Webull 板块涨跌家数、大单资金流向
- 网上搜索的消息面：宏观数据、美联储、AI 资本开支、出口管制、龙头财报、政策

研判的权限由代码限死，**只能让系统更保守**：

| 研判 | 效果 |
|---|---|
| 偏空 | 拦截所有做多买入（反向 ETF 除外） |
| 偏空且置信度 ≥ 0.7 | 允许提议对冲：只限 `hedge.instruments` 名单（默认 SOXS），金额 ≤ 本金 3%，最多持有 10 天，ATR 止损，研判转向后自动平仓 |
| 中性 / 偏多 | 不放宽任何规则 |

每天的研判都记在 `decisions/<日期>/decisions.json` 和日志里，仪表盘显示历史，方便事后检验判断准不准。**这部分没法回测**（模型知道历史），只能靠前向记录检验。

## 事件日历

- **宏观事件**写在 `events.yaml`（CPI、非农、FOMC 等，附市场预期和来源），Claude 每天审核时维护；**财报日期**自动从 Webull 获取
- **事件前推演**：重大事件公布前，Claude 在市场研判里写三种情景（强于 / 符合 / 弱于预期）的触发条件、可能的市场反应和系统应对。不预测具体数字
- **事件后复盘**：公布后的第一次审核，对照推演写实际值、市场反应、命中了哪个情景
- **代码执行的规则**（`config.yaml` → `events`）：重大宏观事件在 24 小时内公布时，新开仓位 ×0.5；某只股票 3 个交易日内发财报时，不买这只。卖出和对冲不受影响

## 日内做 T 回测（研究用，未接入交易）

```bash
.venv/bin/python -m wquant intraday-bt --fetch   # 下载最近 180 天 5 分钟线并回测
```

用 SOXL（做多）/ SOXS（做空）的常规时段 5 分钟线，测试开盘区间突破、VWAP 回归（含 / 不含 QQQ 过滤）、EMA 趋势跟随。
信号在 K 线收盘产生、下一根开盘成交，计滑点和止损，当天收盘全部平仓。

**2026-05 到 2026-11 的结果：四种策略全部亏损**；30 组参数里没有一组前后两段都赚钱；只看高波动日、或者按日线方向只做一边，也都亏。
所以目前**没有**把日内交易接入自动下单。分钟数据会继续积累，以后有新想法可以用同一套框架验证。

注意：分钟线没有做拆股 / 合股调整（SOXS 2026-07-15 有 1:10 合股），日内回测不持仓过夜所以不受影响，但不要用分钟线算跨日收益。

## 夜盘方向策略回测（研究中，未接入交易）

```bash
.venv/bin/python -m wquant overnight-bt --fetch            # 默认：04:00 ET 判断，≥3% 顺势，止损 6%，可买回，收盘平仓
.venv/bin/python -m wquant overnight-bt --korea 1          # 加韩国（EWY）同向确认
```

思路：美东 04:00（北京时间 17:00，夜盘结束）看 SOXL 相对前一天收盘的涨跌，超过阈值就顺势买 SOXL 或 SOXS，持有到当天收盘；
止损 6%，止损后从最低点反弹 3% 可以按原方向买回一次。

2026-05 到 2026-11（124 天）的结果：+$10,678（每笔 $1 万，胜率 51%，PF 1.46，最大回撤 3.1%）；
288 组参数全部赚钱；方向反着做 −$19,239，随机方向平均 −$1,668。
**主要风险**：大部分利润来自做空一侧（SOXS），而这段时间 SOXL 跌了 74%；后半段明显变弱（+$1,762），10 月小亏。
换到 TQQQ/SQQQ 也赚，但同样主要来自做空一侧。需要在沙盒里前向测试，并在上涨行情里再验证。

## 半导体短线系统（SOXL / SOXS）—— 当前在沙盒模拟

每个美股交易日：

| 时间（北京，冬令时） | 谁 | 做什么 |
|---|---|---|
| 08:00 左右 | Claude（`/trade-review`） | 写市场研判：新闻、战争、油价、利率、宏观数据、事件 → 给出半导体方向 `semis.bias`（long / short / both / none）和置信度 |
| 08:45 | 程序（launchd 自动启动） | 战术层监控进程启动 |
| 10:00 / 12:00 / 14:30（韩国收盘）/ 17:00 | 程序 | 检查：SOXL 相对昨收 ≥±3% **且 KOSPI 综合指数**当天同方向 ≥1% **且** 符合研判方向（置信度 ≥0.7 时才限制）→ 买 SOXL 或 SOXS |
| 持仓期间每分钟 | 程序 | 止损：跌破均价 6%；盘前 / 常规时段每比上次买入价再涨 3% 加仓 33%（最多 130%，夜盘不融资最多 100%）；加过仓后从最高价回撤 6% 止盈；卖出后从低点反弹 3% 按原方向买回，不限次数 |
| 次日 04:55 | 程序 | 全部平仓，不持仓过夜 |

- 系统本金 $3,000 独立记账，**初始仓位** = 33% × 信号强弱倍数（1～2），即 33%～66%
  - 信号强弱倍数 = min(SOXL 相对昨收涨跌 / 3%, KOSPI 涨跌 / 1%)，全部是当天实时数据，没有滞后
  - 不用过去几天的趋势 / 波动率指标：有 4～6 天滞后，还会压缩趋势反转时（表现最好）的交易
- 回测（2026-05 ～ 11，124 天）：+184%，最大回撤 -18%，最差一天 -11.7%（夜盘滑点按 0.3% 算 +160%）。
  对比：固定 33% 是 +51% / -9.6%；只按信号强弱、不加仓是 +101% / -12%。加仓多赚的钱主要来自少数大单边日。
  持仓过夜的版本全部亏损所以不用
- 融资：需要账户对 SOXL / SOXS 有融资额度（杠杆 ETF 的保证金要求通常更高，请向 Webull 确认）
- KOSPI 来自 Yahoo Finance 公开行情；韩国休市日拿不到当天数据就不交易
- 研判过滤这一层没法回测，靠这段模拟期验证
- 日线长周期核心层（`swing.core`）目前关闭

```bash
.venv/bin/python -m wquant swing status                    # 系统权益、持仓、今天的记录
.venv/bin/python -m wquant swing tactical                  # 手动跑当天（dry_run，只打印）
.venv/bin/python -m wquant swing tactical --mode webull    # 手动跑当天（沙盒下单）
deploy/install.sh                                          # 安装定时任务（战术层 + 仪表盘）
deploy/uninstall.sh                                        # 移除定时任务
```

## 仪表盘

```bash
.venv/bin/python -m wquant dashboard      # 打开 http://127.0.0.1:8765
```

只在本机监听，只读，不会下单。显示：系统权益曲线、持仓盈亏和止损位、每天的候选和 Claude 审核理由、Webull 订单、规则层回测和基准对比。每 60 秒自动刷新；权益每 10 分钟记录一次（仪表盘开着时才会记录）。

## 关于审核层能不能回测

**不能可靠地回测。** Claude 的训练数据里已经包含了这些历史行情和新闻，拿它去"回测" 2023 年的决策，等于让它看着答案考试，结果会严重高估。所以：

1. 回测只验证**规则层**（`wquant backtest`）
2. 审核层在**沙盒里前向测试**：每天跑一遍，积累几周到几个月的决策日志
3. 对比被否决的那些候选后来的走势，判断审核到底是在帮忙还是在添乱

`logs/*.jsonl` 里每一条都记录了规则理由、AI 的决定、置信度和风险点，方便做这种复盘。

## 定时运行

日线策略每天跑一次就够。建议美东时间 9:45（开盘 15 分钟后；北京时间冬令时 22:45、夏令时 21:45），用前一天的收盘数据下 DAY 限价单。

审核需要 Claude Code 参与，有两种方式：
- **手动**：每天打开 Claude Code，输入 `/trade-review`
- **自动**：在 Claude 桌面应用里建一个定时任务（工作日北京时间 22:45，美国夏令时期间改为 21:45），内容是"在 ~/webull-quant 运行 /trade-review"。沙盒环境下它会一路执行到下单；实盘环境下它只写决策，最后一步由你自己运行

## 主要配置（config.yaml）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `risk.max_position_pct` | 0.10 | 单只股票最多占总资产的比例 |
| `risk.max_order_notional` | 5000 | 单笔订单金额上限，会和上一项一起生效，取较小值 |
| `risk.max_daily_loss_pct` | 0.03 | 当日亏损超过这个比例就停止开新仓 |
| `strategy.market_filter` | SPY/QQQ 对 SMA200 | 任一指数跌破均线就停止开新仓，只执行卖出和止损 |
| `risk.max_capital` | 100000 | 系统可动用的本金上限（美元） |
| `order.trading_session` | ALL_DAY | 交易时段；非常规时段自动用限价单 |
| `ai.mode` | claude_code | 审核方式，见上表 |
| `ai.min_confidence` | 0.6 | 批准买入时的最低置信度 |
| `order.type` | LIMIT | 限价单，价格 = 最新价 ± `limit_offset_bps` |

## 和官方 MCP / skills 怎么配合

这个系统直接调用官方 Python SDK（`webull-openapi-python-sdk`），下单的参数格式参照了官方 `webull-openapi-mcp`。

官方 MCP 更适合**人和 Claude 一起用**，比如查持仓、研究个股、复盘日志，可以另外配到 Claude Desktop 或 Claude Code 里。建议给 MCP 配上 `WEBULL_TOOLSETS=account,instrument,market-data`（不带交易工具）和 `WEBULL_SYMBOL_WHITELIST`，把自动下单只留给这个系统，责任清楚。

## 免责声明

本项目只是工程框架，示例策略不构成任何投资建议。上实盘之前，请先在沙盒里充分验证，并自己承担交易风险。
