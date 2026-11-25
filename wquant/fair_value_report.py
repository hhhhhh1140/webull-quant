"""每日开盘前公允价值报告生成（终端表格 + Markdown + JSON）与回测汇总打印。"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .config import ROOT
from .fair_value import StockFairValue

DECISIONS_DIR = ROOT / "decisions"


def _fmt_price(v: float | None) -> str:
    return f"${v:,.2f}" if v is not None else "-"


def _fmt_pct(v: float | None, signed: bool = True) -> str:
    if v is None:
        return "-"
    return f"{v:+.2f}%" if signed else f"{v:.2f}%"


def _fmt_num(v: float | None, digits: int = 2, signed: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def _src_tag(src: str) -> str:
    return {
        "overnight": "夜盘",
        "pre_market": "盘前",
        "post_market": "盘后",
        "regular": "现价",
        "prev_close": "昨收",
        "close": "收盘",
        "live": "实时",
    }.get(src, src)


def render_markdown_report(
    results: list[StockFairValue],
    event_context: dict[str, Any] | None = None,
    regime_context: dict[str, Any] | None = None,
    report_date: str | None = None,
    source: str = "yahoo",
    backtest_summary: dict[str, Any] | None = None,
) -> str:
    """生成开盘前/夜盘实时公允价值 Markdown 报告。"""
    day = report_date or str(date.today())
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ev = event_context or {}
    rg = regime_context or {}
    macro_alerts = ev.get("macro_alerts") or []
    earnings_flags = ev.get("earnings_flags") or {}

    under = [r for r in results if r.z_score <= -0.35]
    over = [r for r in results if r.z_score >= 0.35]
    neutral = [r for r in results if -0.35 < r.z_score < 0.35]

    lines: list[str] = [
        f"# 每日美股公允价值报告（{day}）",
        "",
        f"- **生成时间**：{now_str} | **行情源**：`{source}` | **覆盖标的**：{len(results)} 只",
        f"- **分布概览**：偏低/低估 {len(under)} 只 · 合理区间 {len(neutral)} 只 · 偏高/高估 {len(over)} 只",
    ]
    if rg.get("summary"):
        lines.append(f"- **市场趋势与跨市场状态**：{rg['summary']}")
    if macro_alerts:
        lines.append(f"- **⚠ 宏观事件提醒**：{'; '.join(macro_alerts)}")
    if earnings_flags:
        ef_str = ", ".join(
            f"{sym} ({info['date']}, {'即将发布' if info['status'] == 'EARNINGS_SOON' else '刚落地'})"
            for sym, info in sorted(earnings_flags.items())
        )
        lines.append(f"- **⚠ 财报窗口（已自动调整回归权重）**：{ef_str}")

    lines.extend([
        "",
        "## 一、当日综合公允价总览（按偏离 Z-score 排序，偏低在前）",
        "",
        "> 偏离度 = `(参考价 / 综合公允价 - 1)`，**负值表示现价低于公允价（偏低）**，正值表示偏高；若含 `BEAR_REGIME` / `KOSPI_SHOCK`，代表处于下跌趋势或亚太半导体急跌冲击，已自动抑制滞后基本面看多拉回。",
        "",
        "| 标的 | 板块 | 参考价 (来源/较昨收) | 综合公允价 | 偏离度 | Z值 | 当日 1σ 合理区间 | 信号 | 主要归因 / 风险提示 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])

    for r in results:
        flag_txt = f" `[{','.join(r.flags)}]`" if r.flags else ""
        lines.append(
            f"| **{r.symbol}** | {r.group} | {_fmt_price(r.ref_price)} ({_src_tag(r.price_source)} {_fmt_pct(r.overnight_chg_pct)}) "
            f"| **{_fmt_price(r.fair_value)}** | {_fmt_pct(r.mispricing_pct)} | {_fmt_num(r.z_score, 2, True)} "
            f"| {_fmt_price(r.low_1s)} ~ {_fmt_price(r.high_1s)} | {r.signal} | {r.attribution}{flag_txt} |"
        )

    lines.extend([
        "",
        "## 二、三层定价明细拆解",
        "",
        "| 标的 | 第1层·板块联动价 (R² / 半衰期) | 第2层·量价均衡价 (预期) | 第3层·基本面锚 (空间) | Fwd P/E (同行中位) | 分析师目标中位 (人数) | 90% 边界区间 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for r in results:
        hl_txt = f"{r.peer_half_life:.0f}d" if r.peer_half_life else "-"
        r2_txt = f"{r.peer_r2:.2f}" if r.peer_r2 is not None else "-"
        l1 = f"{_fmt_price(r.peer_fv)} ({r2_txt}/{hl_txt})" if r.peer_fv else "-"
        l2 = f"{_fmt_price(r.stat_fv)} ({_fmt_pct(r.stat_pred_ret_pct)})" if r.stat_fv else "-"
        l3 = f"{_fmt_price(r.fund_fv)} ({_fmt_pct(r.fund_gap_pct)})" if r.fund_fv else "-"
        pe_txt = (
            f"{r.forward_pe:.1f} ({r.peer_median_pe:.1f})"
            if r.forward_pe is not None and r.peer_median_pe is not None
            else (f"{r.forward_pe:.1f}" if r.forward_pe is not None else "-")
        )
        tgt_txt = f"{_fmt_price(r.target_median)} ({r.analyst_count})" if r.target_median else "-"
        band90 = f"{_fmt_price(r.low_90)} ~ {_fmt_price(r.high_90)}"
        lines.append(f"| **{r.symbol}** | {l1} | {l2} | {l3} | {pe_txt} | {tgt_txt} | {band90} |")

    if backtest_summary and not backtest_summary.get("error"):
        lines.extend(["", "## 三、历史滚动样本外验证（Walk-Forward）", ""])
        lines.append(format_backtest_markdown(backtest_summary))

    lines.append("")
    return "\n".join(lines)


def format_terminal_table(
    results: list[StockFairValue],
    event_context: dict[str, Any] | None = None,
    regime_context: dict[str, Any] | None = None,
) -> str:
    """生成适合终端直接阅读的对齐表格。"""
    ev = event_context or {}
    rg = regime_context or {}
    lines: list[str] = []
    if rg.get("summary"):
        lines.append(f"★ 市场状态: {rg['summary']}")
    if ev.get("macro_alerts"):
        lines.append(f"⚠ 宏观事件: {'; '.join(ev['macro_alerts'])}")
    if ev.get("earnings_flags"):
        ef = ", ".join(f"{k}({v['date']})" for k, v in sorted(ev["earnings_flags"].items()))
        lines.append(f"⚠ 财报窗口: {ef}")

    header = (
        f"{'标的':<6} {'板块':<9} {'参考价':>9} {'较昨收':>8} {'公允价':>9} {'偏离':>8} {'Z值':>6} "
        f"{'1σ合理区间':>19} {'L1联动':>9} {'L2量价':>9} {'L3基本面锚':>10} {'信号':<6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        rng = f"{r.low_1s:>7.2f}~{r.high_1s:<7.2f}"
        l1 = f"{r.peer_fv:9.2f}" if r.peer_fv is not None else f"{'-':>9}"
        l2 = f"{r.stat_fv:9.2f}" if r.stat_fv is not None else f"{'-':>9}"
        l3 = f"{r.fund_fv:10.2f}" if r.fund_fv is not None else f"{'-':>10}"
        lines.append(
            f"{r.symbol:<6} {r.group:<9} {r.ref_price:9.2f} {r.overnight_chg_pct:+7.2f}% {r.fair_value:9.2f} "
            f"{r.mispricing_pct:+7.2f}% {r.z_score:+6.2f} {rng:>19} {l1} {l2} {l3} {r.signal}"
        )
    return "\n".join(lines)


def format_backtest_markdown(bt: dict[str, Any]) -> str:
    """格式化 Walk-Forward 回测结果为 Markdown / 终端可读文本。"""
    if bt.get("error"):
        return f"回测失败: {bt['error']}"
    layers = bt["layers"]
    bk = bt["buckets_z1"]
    cal = bt["range_calibration"]
    lines = [
        f"- **回测区间**：{bt['period']}（{bt['trading_days']} 个交易日，{bt['symbols_evaluated']} 只标的，共 {bt['total_samples']} 样本）",
        f"- **区间校准**：次日收盘落入 1σ 区间 **{cal['in_1s_pct']}%**（理论 68.3%），落入 90% 区间 **{cal['in_90_pct']}%**（理论 90.0%）",
        "",
        "| 模型层 | 1日 Rank IC | 3日 Rank IC | 5日 Rank IC | 3日 Pearson IC |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for name, label in (("peer", "第1层·板块联动"), ("stat", "第2层·量价均衡"), ("composite", "综合融合 (L1+L2)")):
        m = layers[name]
        lines.append(
            f"| {label} | {m['rank_ic_1d']:+.4f} | {m['rank_ic_3d']:+.4f} | {m['rank_ic_5d']:+.4f} | {m['ic_3d']:+.4f} |"
        )

    lines.extend([
        "",
        f"- **五分位多空分组（最被低估 Top 20% vs 最被高估 Bottom 20%）**：",
        f"  - 低估组 (Z ≤ {bk.get('q20_z_threshold', -0.35):+.2f}，{bk['undervalued_count']} 次)：未来 3 日均值 **{_fmt_pct(bk['undervalued_avg_ret_3d_pct'])}**，上涨胜率 **{_fmt_pct(bk['undervalued_hit_3d_pct'], False)}**",
        f"  - 高估组 (Z ≥ {bk.get('q80_z_threshold', 0.35):+.2f}，{bk['overvalued_count']} 次)：未来 3 日均值 **{_fmt_pct(bk['overvalued_avg_ret_3d_pct'])}**，下跌胜率 **{_fmt_pct(bk['overvalued_hit_3d_pct'], False)}**",
        f"  - 多空收益差（低估 − 高估）：3日 **{_fmt_pct(bk['long_short_spread_3d_pct'])}** · 5日 **{_fmt_pct(bk['long_short_spread_5d_pct'])}**",
        "",
        "| 板块 | 样本数 | 3日 Rank IC | 5日 Rank IC | 3日多空差 (五分位) | 1σ覆盖率 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for grp, g in sorted(bt["by_group"].items()):
        lines.append(
            f"| {grp} | {g['samples']} | {g['rank_ic_3d']:+.4f} | {g['rank_ic_5d']:+.4f} "
            f"| {_fmt_pct(g['ls_spread_3d_pct'])} | {g['in_1s_pct']:.1f}% |"
        )
    return "\n".join(lines)


def save_daily_report(
    results: list[StockFairValue],
    event_context: dict[str, Any] | None = None,
    regime_context: dict[str, Any] | None = None,
    report_date: str | None = None,
    source: str = "yahoo",
    backtest_summary: dict[str, Any] | None = None,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """将每日公允价值报告写入 decisions/<日期>/fair_value.json 与 fair_value.md。"""
    day = report_date or str(date.today())
    target_dir = out_dir or (DECISIONS_DIR / day)
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / "fair_value.json"
    md_path = target_dir / "fair_value.md"

    payload = {
        "date": day,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "regime": regime_context or {},
        "events": event_context or {},
        "count": len(results),
        "items": [r.to_dict() for r in results],
        "backtest": backtest_summary,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_text = render_markdown_report(
        results,
        event_context=event_context,
        regime_context=regime_context,
        report_date=day,
        source=source,
        backtest_summary=backtest_summary,
    )
    md_path.write_text(md_text, encoding="utf-8")
    return json_path, md_path
