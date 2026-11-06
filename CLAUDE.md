# webull-quant

美股/ETF 日线量化系统，基于 Webull 官方 Python SDK。Claude Code 是系统的审核层。

## 每日流程

1. `python -m wquant scan` → 规则层生成 `decisions/<日期>/candidates.json`
2. Claude Code 审核（`/trade-review` 技能）→ 先写市场研判和对冲提议，再审核个股 → `decisions/<日期>/decisions.json`
3. `python -m wquant execute` → 风控 → 下单 → `logs/<日期>.jsonl`

所有命令都用 `.venv/bin/python -m wquant ...`。

## 规则

- 程序默认不调用 Claude API（`ai.mode: claude_code`）。不要把模式改成 `api`，除非用户明确要求。
- WEBULL_ENVIRONMENT=prod 时，Claude 不运行 `execute`、不下单，只把命令交给用户。
- 不要读取或打印 `.env`，不要把密钥写进任何文件或输出。
- 风控参数（config.yaml 的 `risk`）和 `KILL_SWITCH` 文件只在用户明确要求时修改。
- 改代码后运行 `.venv/bin/python -m pytest -q`。

## 结构

- `wquant/strategy.py` 规则信号 · `analyst.py` 决策格式 · `risk.py` 风控
- `wquant/engine.py` scan / execute · `broker.py` 执行器 · `webull_client.py` SDK 封装
- `wquant/backtest.py` 回测（只回测规则层）
