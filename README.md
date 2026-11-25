# Webull 半导体双层量化策略

## 回测总结果

| 样本 | 期末权益 | 累计收益 | 最大日度回撤 | 年化夏普 |
|---|---:|---:|---:|---:|
| 2020-01-02 至 2026-11-24 | $35,355.72 | **+1,078.52%** | -74.79% | 0.86 |
| 2026-01-02 至 2026-11-24 | $20,664.44 | **+588.81%** | -41.30% | 2.44 |

两段各以 **$3,000** 起算，只交易 **SOXL / SOXS**。纯规则版按 SOXX 三票在核心层和夜盘战术层之间切换，每次使用可用资金买整股，无融资、无加仓。

[回测汇总](reports/dual-layer-2026-11-24-rules-only/summary.json) · [2026 月度明细](reports/dual-layer-2026-11-24-rules-only/2026_monthly.csv) · [2026 逐笔模拟成交](reports/dual-layer-2026-11-24-rules-only/2026_ytd_hybrid_trades.csv) · [2020–2026 逐笔模拟成交](reports/dual-layer-2026-11-24-rules-only/2020_2026_hybrid_trades.csv)

2026 年有 123 个使用 1 分钟 ETF 行情的交易日和 13 个使用 5 分钟行情的交易日。缺少分钟行情的日期仅按日线近似核心层。KOSPI 取当时已知的 1 分钟、5 分钟或小时行情。模拟允许卖出后立即复用资金，未扣税费。回测以 K 线触及限价及成交量估算整笔成交。

## 交易规则

SOXX 前一交易日的已完成日线提供三张票：唐奇安 20/10、SMA50 与 20 日动量、EMA10/30。每张票取 +1、0 或 −1。

| 总票数 | 执行方式 |
|---|---|
| +2 至 +3 | 核心层，美东 09:30 后买 SOXL，可以过夜 |
| −1 至 +1 | 平核心仓，战术层按信号买 SOXL 或 SOXS |
| −3 至 −2 | 平核心仓，战术层只允许买 SOXS |

- **核心层**：以本笔入场均价设 12% 固定止损。若上涨趋势延续，自止损后的下一个美股交易日起，价格从当日常规盘低点反弹 5% 时买回。等待期间的战术信号只允许做多。
- **战术进场**：美东 21:00–04:00 每分钟扫描。SOXL 较常规盘昨收同方向变化至少 3%，KOSPI 同方向至少 1%，再检查 6% 路径否决。韩国休市跳过。
- **战术退出和买回**：首腿跌破均价 6% 或从持仓最高价回撤 7% 时退出。卖出后，若在 15:00 前从当日卖出后的低点反弹 3%，可以按原方向买回一次。第二腿的固定止损为买入均价下方 12%。15:55 提交清仓。
- **订单**：使用覆盖当前交易时段的 DAY 限价单。买入限价较最新价上浮 0.3%，卖出限价下浮 0.3%。

## 安装和运行

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m wquant check
```

在 `.env` 填写 Webull 行情与账户配置。项目默认使用 UAT 沙盒配置模板。核心层和战术层共用一个运行入口，交易方向来自程序规则，无需人工每日审核。

```bash
.venv/bin/python -m wquant run                # 演练，只打印订单，使用独立本地演练账本
.venv/bin/python -m wquant run --mode webull  # 调用 .env 配置的 Webull 环境
.venv/bin/python -m wquant status             # 查看双层策略持仓和当日状态
.venv/bin/python -m wquant dashboard          # 只读仪表盘，http://127.0.0.1:8765
```

`swing tactical` 是同一双层运行入口。macOS 可在核对 `deploy/*.plist` 中的安装路径后，用 `deploy/install.sh` 安装沙盒任务与仪表盘。实盘提交同时要求 `WEBULL_ENVIRONMENT=prod`、`live_trading_enabled: true` 和 `--confirm-live`。在项目根目录创建 `KILL_SWITCH` 可以停止新开仓。

运行账本使用下单参考价记录均价，券商部分成交或未成交时可能与实际持仓和成交均价有差异。仪表盘订单表展示券商回报。

## 复跑与研究

```bash
.venv/bin/python -m wquant.dual_layer_replay INPUT_DIR OUTPUT_DIR
.venv/bin/python -m pytest -q
```

`INPUT_DIR` 提供 `SOXX.csv`、`SOXL.csv`、`SOXS.csv`、`KOSPI.csv` 日线。ETF 分钟归档使用 `SOXL_M1.csv.gz`、`SOXS_M1.csv.gz` 及相应的 `M5` 文件，包含时间戳、OHLC、成交量和交易时段。KOSPI 分钟归档使用 `KOSPI_1m_closed.csv.gz`、`KOSPI_5m_closed.csv.gz` 或 `KOSPI_60m_closed.csv.gz`，时间戳为 K 线完成时刻。ETF 拆合股和派息分别通过 `<SYMBOL>_splits.csv` 与 `<SYMBOL>_dividends.csv` 提供。原始行情放在本地缓存目录。

[早期策略与实验](docs/research.md) · [历史对照回测](reports/dual-layer-2026-11-24-cost-stop/summary.json)
