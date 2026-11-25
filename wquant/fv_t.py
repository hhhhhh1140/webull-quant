"""盘中 5 分钟动态公允价反复做 T 引擎（Intraday Dynamic Fair-Value StatArb）。

核心逻辑（华尔街 StatArb 降频版）：
  1. 滚动因子拟合（严格用前 beta_days 个交易日的 5 分钟收益，零前视）：
     对每只股票拟合 r_i ~ beta_qqq * r_QQQ + beta_peer * r_peer（同板块等权篮子）。
  2. 盘中动态公允价（每根收盘 M5 重算一次）：
     - Peer-Implied FV：以过去 anchor_bars 根 M5（如 60 分钟）前为锚，代入 QQQ 与同板块累计涨跌，
       算出“按同行此刻表现，这只股票现在应该在什么价位”。
     - Intraday VWAP：当日累计成交量加权均价（机构执行中枢）。
     - Dynamic FV = exp(w_peer * ln(Peer_FV) + (1 - w_peer) * ln(VWAP))。
  3. 反复做 T（低买高卖）：
     - 买入（低买）：现价低于动态公允价达到 entry_pct 且 Z <= -entry_z；可选 1 根企稳确认（防接飞刀）；
       同时满足 QQQ 盘中未崩盘、未达单股日内止损上限、未达仓位/笔数/日亏熔断上限。
     - 卖出（高卖/风控）：回归到动态公允价附近（FV * (1 + exit_fv_offset)）或达到止盈 target_pct；
       触发硬止损 -stop_pct、超时 max_hold_bars 或尾盘 15:50 ET 强平。
     - 平仓冷却 cooldown_bars 根后可再次低吸，实现盘中反复做 T。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .fv_t_data import split_rth_by_day


@dataclass
class FvTParams:
    capital: float = 10_000.0
    notional_per_trade: float = 2_000.0
    max_positions: int = 4
    max_per_group: int = 2
    max_trades_per_day: int = 16
    max_stops_per_symbol: int = 1      # 单股单日止损达到此次数后，当天不再做这只（防单边阴跌反复割肉）
    daily_loss_pct: float = 0.02       # 当日净亏超过 2% 熔断停止开新仓
    # 因子与动态公允价参数
    beta_days: int = 10                # 用过去 N 个交易日 M5 收益拟合 Beta
    anchor_bars: int = 12              # 12 根 M5 = 60 分钟相对联动窗口
    peer_weight: float = 0.70          # 动态公允价中板块联动权重（其余为当日 VWAP）
    min_peer_r2: float = 0.15          # 低于此拟合优度说明板块联动太弱，跳过
    # 进出场阈值
    entry_pct: float = 0.010           # 低于动态公允价至少 1.0%
    entry_z: float = 1.5               # 且偏离至少 -1.5σ
    confirm_reversal: bool = True      # 要求当前收盘 K 线止跌企稳（防直落接飞刀）
    qqq_crash_pct: float = -0.015      # QQQ 相对今日开盘跌破 -1.5% 时暂停抄底
    exit_fv_offset: float = 0.001      # 回到动态公允价上方 +0.1% 即止盈卖出
    target_pct: float = 0.018          # 硬止盈 +1.8%
    stop_pct: float = 0.015            # 硬止损 -1.5%
    max_hold_bars: int = 18            # 最长持有 18 根 M5（90 分钟）
    cooldown_bars: int = 2             # 平仓后冷却 2 根 M5（10 分钟）再允许重新低吸
    # 交易时段（以 09:30 为第 0 根 M5，共 78 根）
    start_bar: int = 4                 # 09:50 ET 起允许开仓（避开开盘前 20 分钟剧烈噪音）
    last_entry_bar: int = 69           # 15:15 ET 后不再开新仓
    eod_exit_bar: int = 75             # 15:45 ET 收盘（15:50 执行）强平所有持仓
    # 模拟成交成本（单边）
    spread_bps: float = 4.0            # 完整点差 4bp（单边吃一半 2bp）
    slippage_bps: float = 4.0          # 单边滑点 4bp
    fee_per_order: float = 0.10        # 单笔手续费（美元）

    @property
    def fill_drag(self) -> float:
        return (self.slippage_bps + self.spread_bps / 2.0) / 10_000.0


@dataclass
class PeerBetaModel:
    beta_qqq: float
    beta_peer: float
    res_std_bar: float   # 单根 5 分钟特异残差标准差
    r2: float
    peers: list[str] = field(default_factory=list)


@dataclass
class IntradaySnapshot:
    symbol: str
    group: str
    bar_time: str
    price: float
    peer_fv: float
    vwap: float
    fair_value: float
    mispricing_pct: float   # (price / fair_value - 1) * 100，负值=低于公允价
    z_score: float
    r2: float
    confirmed: bool         # 是否出现 1 根企稳回钩
    peers: list[str] = field(default_factory=list)


def fit_intraday_betas(
    day_list: list[date],
    by_day: dict[date, dict[str, pd.DataFrame]],
    target_day_idx: int,
    symbols: list[str],
    groups: dict[str, str],
    beta_days: int = 10,
    ridge_alpha: float = 1.0,
) -> dict[str, PeerBetaModel]:
    """严格用 target_day_idx 之前最近 beta_days 个交易日的 M5 收益拟合每只股票的 (beta_qqq, beta_peer)。"""
    start_i = max(0, target_day_idx - beta_days)
    hist_days = day_list[start_i:target_day_idx]
    if len(hist_days) < 3:
        return {}

    models: dict[str, PeerBetaModel] = {}
    for sym in symbols:
        grp = groups.get(sym, "default")
        peers = [p for p, g in groups.items() if g == grp and p != sym]

        y_list: list[np.ndarray] = []
        x_qqq_list: list[np.ndarray] = []
        x_peer_list: list[np.ndarray] = []
        used_peers: set[str] = set()

        for d in hist_days:
            dmap = by_day[d]
            if sym not in dmap or "QQQ" not in dmap:
                continue
            avail_peers = [p for p in peers if p in dmap]
            if not avail_peers:
                continue
            cols = {"_y": dmap[sym]["close"], "_q": dmap["QQQ"]["close"]}
            for p in avail_peers:
                cols[p] = dmap[p]["close"]
            panel = pd.DataFrame(cols).dropna()
            if len(panel) < 20:
                continue
            log_p = np.log(panel.to_numpy(dtype=float))
            rets = np.diff(log_p, axis=0)
            y_list.append(rets[:, 0])
            x_qqq_list.append(rets[:, 1])
            x_peer_list.append(rets[:, 2:].mean(axis=1))
            used_peers.update(avail_peers)

        if not y_list:
            continue
        y = np.concatenate(y_list)
        if len(y) < 120:
            continue
        X = np.column_stack([np.concatenate(x_qqq_list), np.concatenate(x_peer_list)])
        mu_x = X.mean(axis=0)
        sd_x = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
        Xs = (X - mu_x) / sd_x
        mu_y = float(y.mean())
        yc = y - mu_y
        w_std = np.linalg.solve(Xs.T @ Xs + ridge_alpha * np.eye(2), Xs.T @ yc)
        betas = w_std / sd_x
        y_hat = mu_y + Xs @ w_std
        res = y - y_hat
        ss_tot = float(np.sum((y - mu_y) ** 2))
        r2 = float(np.clip(1.0 - float(np.sum(res ** 2)) / ss_tot, 0.0, 1.0)) if ss_tot > 1e-12 else 0.0
        res_std = max(float(np.std(res, ddof=1)), 0.0012)

        models[sym] = PeerBetaModel(
            beta_qqq=float(np.clip(betas[0], -0.5, 2.5)),
            beta_peer=float(np.clip(betas[1], 0.0, 2.5)),
            res_std_bar=res_std,
            r2=r2,
            peers=sorted(used_peers),
        )
    return models


def compute_intraday_snapshots(
    day_bars: dict[str, pd.DataFrame],
    bar_idx: int,
    symbols: list[str],
    groups: dict[str, str],
    betas: dict[str, PeerBetaModel],
    params: FvTParams,
) -> tuple[dict[str, IntradaySnapshot], float]:
    """在当日第 bar_idx 根已收盘 M5 处，计算全部标的的实时动态公允价与 QQQ 日内涨跌。

    严格只访问 [0 .. bar_idx]，绝不接触未来 K 线。
    """
    qqq_df = day_bars.get("QQQ")
    if qqq_df is None or bar_idx >= len(qqq_df) or bar_idx < 2:
        return {}, 0.0

    ts = qqq_df.index[bar_idx]
    qqq_open = float(qqq_df["open"].iloc[0])
    qqq_cur = float(qqq_df["close"].iloc[bar_idx])
    qqq_day_ret = (qqq_cur / qqq_open - 1.0) if qqq_open > 0 else 0.0

    anchor = max(0, bar_idx - params.anchor_bars)
    span_bars = max(1, bar_idx - anchor)
    qqq_anchor = float(qqq_df["close"].iloc[anchor])
    qqq_move = float(np.log(qqq_cur / qqq_anchor)) if qqq_anchor > 0 else 0.0

    snaps: dict[str, IntradaySnapshot] = {}
    for sym in symbols:
        m = betas.get(sym)
        df = day_bars.get(sym)
        if m is None or df is None or bar_idx >= len(df) or m.r2 < params.min_peer_r2:
            continue
        cur_p = float(df["close"].iloc[bar_idx])
        anc_p = float(df["close"].iloc[anchor])
        if cur_p <= 0 or anc_p <= 0:
            continue

        # 同板块同行在同一窗口 [anchor -> bar_idx] 的等权对数涨跌
        peer_moves: list[float] = []
        active_peers: list[str] = []
        for p in m.peers:
            pdf = day_bars.get(p)
            if pdf is not None and bar_idx < len(pdf):
                p0, p1 = float(pdf["close"].iloc[anchor]), float(pdf["close"].iloc[bar_idx])
                if p0 > 0 and p1 > 0:
                    peer_moves.append(float(np.log(p1 / p0)))
                    active_peers.append(p)
        if not peer_moves:
            continue

        peer_move = float(np.mean(peer_moves))
        implied_ret = m.beta_qqq * qqq_move + m.beta_peer * peer_move
        peer_fv = float(anc_p * np.exp(np.clip(implied_ret, -0.15, 0.15)))

        # 今日累计 VWAP（严格截至 bar_idx）
        sl = df.iloc[: bar_idx + 1]
        tp = (sl["high"] + sl["low"] + sl["close"]) / 3.0
        vol_sum = float(sl["volume"].sum())
        vwap = float((tp * sl["volume"]).sum() / vol_sum) if vol_sum > 0 else float(sl["close"].mean())

        wp = float(np.clip(params.peer_weight, 0.0, 1.0))
        dyn_fv = float(np.exp(wp * np.log(peer_fv) + (1.0 - wp) * np.log(vwap)))
        mis_log = float(np.log(cur_p / dyn_fv))
        mis_pct = (cur_p / dyn_fv - 1.0) * 100.0
        z = float(mis_log / (m.res_std_bar * np.sqrt(span_bars)))

        # 企稳回钩确认：当前 K 线收阳（close > open）或高于前一根收盘，且前一根收盘也低于当前公允价或正在低位
        prev_close = float(df["close"].iloc[bar_idx - 1])
        cur_open = float(df["open"].iloc[bar_idx])
        confirmed = (cur_p > cur_open) or (cur_p > prev_close)

        snaps[sym] = IntradaySnapshot(
            symbol=sym,
            group=groups.get(sym, "default"),
            bar_time=ts.isoformat(),
            price=round(cur_p, 4),
            peer_fv=round(peer_fv, 4),
            vwap=round(vwap, 4),
            fair_value=round(dyn_fv, 4),
            mispricing_pct=round(mis_pct, 3),
            z_score=round(z, 3),
            r2=round(m.r2, 3),
            confirmed=bool(confirmed),
            peers=active_peers,
        )
    return snaps, qqq_day_ret


@dataclass
class FvTTrade:
    day: str
    symbol: str
    group: str
    qty: int
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    entry_fv: float
    entry_z: float
    hold_bars: int
    pnl: float
    ret_pct: float
    reason: str


def run_fv_t_backtest(
    bars_map: dict[str, pd.DataFrame],
    symbols: list[str],
    groups: dict[str, str],
    params: FvTParams | None = None,
) -> dict[str, Any]:
    """严格逐日逐 5 分钟 K 线回放（bar t 收盘信号 → bar t+1 开盘成交，含止损盘中检查与完整成本）。"""
    p = params or FvTParams()
    by_day = split_rth_by_day(bars_map)
    day_list = list(by_day.keys())
    if len(day_list) <= p.beta_days + 2:
        return {"error": f"有效交易日不足（当前 {len(day_list)} 天，至少需要 {p.beta_days + 3} 天）"}

    drag = p.fill_drag
    fee = p.fee_per_order
    equity = p.capital
    daily_records: list[dict[str, Any]] = []
    all_trades: list[FvTTrade] = []

    for d_idx in range(p.beta_days, len(day_list)):
        day = day_list[d_idx]
        raw_day_bars = by_day[day]
        if "QQQ" not in raw_day_bars:
            continue

        # 以 QQQ 时间轴对齐当日所有标的（避免缺根错位）
        q_idx = raw_day_bars["QQQ"].index
        if len(q_idx) < 40:
            continue
        aligned: dict[str, pd.DataFrame] = {"QQQ": raw_day_bars["QQQ"]}
        for s in symbols:
            if s in raw_day_bars:
                sub = raw_day_bars[s].reindex(q_idx)
                if sub["close"].isna().sum() <= 2:
                    aligned[s] = sub.ffill().bfill()

        betas = fit_intraday_betas(day_list, by_day, d_idx, symbols, groups, beta_days=p.beta_days)
        if not betas:
            continue

        day_start_eq = equity
        cash = equity
        day_pnl = 0.0
        day_entries = 0
        halted = False
        positions: dict[str, dict[str, Any]] = {}
        cooldown_until: dict[str, int] = {}
        stops_today: dict[str, int] = {}

        n_bars = len(q_idx)
        eod_bar = min(p.eod_exit_bar, n_bars - 2)
        last_entry = min(p.last_entry_bar, eod_bar - 2)

        def _close_pos(sym: str, exec_bar: int, raw_exit: float, reason: str) -> None:
            nonlocal cash, day_pnl, halted
            pos = positions.pop(sym)
            fill = raw_exit * (1.0 - drag)
            trade_pnl = pos["qty"] * (fill - pos["entry_fill"]) - 2.0 * fee
            cash += pos["qty"] * fill - fee
            day_pnl += trade_pnl
            ret_pct = (fill / pos["entry_fill"] - 1.0) * 100.0
            all_trades.append(
                FvTTrade(
                    day=str(day),
                    symbol=sym,
                    group=pos["group"],
                    qty=pos["qty"],
                    entry_time=q_idx[pos["entry_bar"]].isoformat(),
                    exit_time=q_idx[exec_bar].isoformat(),
                    entry_price=round(pos["entry_fill"], 4),
                    exit_price=round(fill, 4),
                    entry_fv=round(pos["entry_fv"], 4),
                    entry_z=round(pos["entry_z"], 2),
                    hold_bars=exec_bar - pos["entry_bar"],
                    pnl=round(trade_pnl, 2),
                    ret_pct=round(ret_pct, 3),
                    reason=reason,
                )
            )
            cooldown_until[sym] = exec_bar + p.cooldown_bars
            if reason == "stop":
                stops_today[sym] = stops_today.get(sym, 0) + 1
            if day_pnl <= -day_start_eq * p.daily_loss_pct:
                halted = True

        for t in range(p.start_bar, eod_bar + 1):
            snaps, qqq_ret = compute_intraday_snapshots(aligned, t, symbols, groups, betas, p)

            # 1) 检查已持仓是否触发止损/止盈/公允价回归/超时/尾盘强平（在下一根 t+1 开盘或盘中触发价成交）
            for sym in list(positions.keys()):
                pos = positions[sym]
                df = aligned[sym]
                bar_low = float(df["low"].iloc[t])
                bar_high = float(df["high"].iloc[t])
                bar_close = float(df["close"].iloc[t])
                next_open = float(df["open"].iloc[t + 1])

                stop_px = pos["raw_entry"] * (1.0 - p.stop_pct)
                target_px = pos["raw_entry"] * (1.0 + p.target_pct)
                held = t - pos["entry_bar"]

                # 保守原则：同一根 K 线同时触及止损和止盈，按先止损计
                if bar_low <= stop_px:
                    _close_pos(sym, t, min(stop_px, float(df["open"].iloc[t])), "stop")
                    continue
                if bar_high >= target_px:
                    _close_pos(sym, t, max(target_px, float(df["open"].iloc[t])), "target")
                    continue

                if t >= eod_bar:
                    _close_pos(sym, t + 1, next_open, "eod")
                    continue

                snap = snaps.get(sym)
                if snap and bar_close >= snap.fair_value * (1.0 + p.exit_fv_offset):
                    _close_pos(sym, t + 1, next_open, "revert_fv")
                    continue
                if held >= p.max_hold_bars:
                    _close_pos(sym, t + 1, next_open, "timeout")
                    continue

            # 2) 扫描低估开仓机会（优先买偏离 Z 值最深且已企稳的标的）
            if halted or t > last_entry or day_entries >= p.max_trades_per_day or qqq_ret <= p.qqq_crash_pct:
                continue
            if len(positions) >= p.max_positions:
                continue

            candidates: list[IntradaySnapshot] = []
            for sym, snap in snaps.items():
                if sym in positions:
                    continue
                if cooldown_until.get(sym, -1) > t:
                    continue
                if stops_today.get(sym, 0) >= p.max_stops_per_symbol:
                    continue
                if snap.mispricing_pct > -p.entry_pct * 100.0 or snap.z_score > -p.entry_z:
                    continue
                if p.confirm_reversal and not snap.confirmed:
                    continue
                candidates.append(snap)

            candidates.sort(key=lambda s: s.z_score)
            for snap in candidates:
                if len(positions) >= p.max_positions or day_entries >= p.max_trades_per_day:
                    break
                grp_cnt = sum(1 for pos in positions.values() if pos["group"] == snap.group)
                if grp_cnt >= p.max_per_group:
                    continue
                raw_entry = float(aligned[snap.symbol]["open"].iloc[t + 1])
                entry_fill = raw_entry * (1.0 + drag)
                qty = int(min(p.notional_per_trade, cash - fee) // entry_fill)
                if qty <= 0:
                    continue
                cash -= qty * entry_fill + fee
                positions[snap.symbol] = {
                    "group": snap.group,
                    "qty": qty,
                    "raw_entry": raw_entry,
                    "entry_fill": entry_fill,
                    "entry_bar": t + 1,
                    "entry_fv": snap.fair_value,
                    "entry_z": snap.z_score,
                }
                day_entries += 1

        # 兜底：若还有残余持仓，用当日最后一根收盘平掉
        for sym in list(positions.keys()):
            _close_pos(sym, n_bars - 1, float(aligned[sym]["close"].iloc[-1]), "eod")

        equity = day_start_eq + day_pnl
        daily_records.append(
            {
                "day": str(day),
                "pnl": round(day_pnl, 2),
                "ret_pct": round(day_pnl / day_start_eq * 100.0, 3),
                "equity": round(equity, 2),
                "trades": day_entries,
                "halted": halted,
            }
        )

    if not daily_records:
        return {"error": "回测区间内没有产生有效交易日"}

    return _summarize_backtest(daily_records, all_trades, p)


def _summarize_backtest(
    daily_records: list[dict[str, Any]],
    trades: list[FvTTrade],
    p: FvTParams,
) -> dict[str, Any]:
    eq_series = pd.Series([p.capital] + [r["equity"] for r in daily_records], dtype=float)
    daily_rets = pd.Series([r["ret_pct"] / 100.0 for r in daily_records], dtype=float)
    total_ret = float(eq_series.iloc[-1] / p.capital - 1.0)
    max_dd = float((eq_series / eq_series.cummax() - 1.0).min())
    sharpe = (
        float(daily_rets.mean() / daily_rets.std(ddof=1) * np.sqrt(252.0))
        if len(daily_rets) > 2 and daily_rets.std(ddof=1) > 1e-12
        else 0.0
    )

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

    # 分半段检验（前半 vs 后半，检查策略稳定性而非单段行情运气）
    mid = len(daily_records) // 2
    first_half_pnl = round(sum(r["pnl"] for r in daily_records[:mid]), 2)
    second_half_pnl = round(sum(r["pnl"] for r in daily_records[mid:]), 2)

    by_reason: dict[str, dict[str, Any]] = {}
    for t in trades:
        b = by_reason.setdefault(t.reason, {"count": 0, "pnl": 0.0})
        b["count"] += 1
        b["pnl"] = round(b["pnl"] + t.pnl, 2)

    by_group: dict[str, dict[str, Any]] = {}
    for t in trades:
        b = by_group.setdefault(t.group, {"count": 0, "wins": 0, "pnl": 0.0})
        b["count"] += 1
        b["wins"] += int(t.pnl > 0)
        b["pnl"] = round(b["pnl"] + t.pnl, 2)
    for g in by_group.values():
        g["win_rate_pct"] = round(g["wins"] / g["count"] * 100.0, 1) if g["count"] else 0.0

    return {
        "period": f"{daily_records[0]['day']} → {daily_records[-1]['day']}",
        "days": len(daily_records),
        "initial_capital": p.capital,
        "final_equity": round(float(eq_series.iloc[-1]), 2),
        "total_pnl": round(float(eq_series.iloc[-1] - p.capital), 2),
        "total_return_pct": round(total_ret * 100.0, 2),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "sharpe": round(sharpe, 2),
        "trades": len(trades),
        "avg_trades_per_day": round(len(trades) / len(daily_records), 2),
        "win_rate_pct": round(len(wins) / len(trades) * 100.0, 1) if trades else 0.0,
        "profit_factor": pf,
        "avg_pnl_per_trade": round(sum(t.pnl for t in trades) / len(trades), 2) if trades else 0.0,
        "avg_hold_mins": round(float(np.mean([t.hold_bars * 5 for t in trades])), 1) if trades else 0.0,
        "split_half_pnl": {"first_half": first_half_pnl, "second_half": second_half_pnl},
        "by_reason": by_reason,
        "by_group": by_group,
        "recent_trades": [asdict(t) for t in trades[-10:]],
    }


# ---------------- 实证有效的 1~5 日动态公允价波段做 T（Swing-T）回测 ----------------

@dataclass
class SwingTParams:
    capital: float = 10_000.0
    notional_per_trade: float = 2_500.0
    max_positions: int = 4
    max_per_group: int = 2
    entry_z: float = -0.60            # 低于动态公允价至少 0.60σ 时低吸
    exit_z: float = 0.10              # 回归到动态公允价上方 +0.10σ 时高卖止盈
    stop_pct: float = 0.06            # 单笔硬止损 -6%
    max_hold_days: int = 5            # 最长持有 5 个交易日
    require_soxx_above: int = -2      # SOXX 三票制 <= -2（熊市趋势）时清仓/停买，避免接飞刀
    kospi_crash_pct: float = -2.0     # KOSPI 当天暴跌 <= -2% 时暂停新开多单
    spread_bps: float = 4.0
    slippage_bps: float = 4.0
    fee_per_order: float = 0.10
    eval_days: int = 500
    retrain_every: int = 5

    @property
    def fill_drag(self) -> float:
        return (self.slippage_bps + self.spread_bps / 2.0) / 10_000.0


def run_swing_t_backtest(
    daily_bars: dict[str, pd.DataFrame],
    symbols: list[str],
    groups: dict[str, str],
    params: SwingTParams | None = None,
) -> dict[str, Any]:
    """严格滚动样本外（Walk-Forward）1~5 日公允价做 T 回测。

    - 收盘后严格用当日及之前数据计算第 1 层板块联动 + 第 2 层短期量价公允价与 Z 值
    - 次日开盘成交（含点差 + 滑点 + 手续费）
    - 叠加 SOXX 三票制下跌趋势过滤（-2 / -3 票自动清仓防接飞刀）
    """
    from .fair_value import build_stat_features, estimate_peer_layer
    from .indicators import atr
    from .swing import ensemble_votes

    p = params or SwingTParams()
    syms = [s.upper() for s in symbols if s.upper() in daily_bars]
    if not syms:
        return {"error": "没有可用日线标的"}

    close_panel = pd.DataFrame({k: v["close"].astype(float) for k, v in daily_bars.items() if len(v) > 0}).sort_index()
    bench_close = close_panel["QQQ"] if "QQQ" in close_panel.columns else close_panel.get("SPY")

    pre_feats = {s: build_stat_features(daily_bars[s], bench_close) for s in syms}
    pre_sigma: dict[str, pd.Series] = {}
    for s in syms:
        cs = daily_bars[s]["close"].astype(float)
        lr = np.log(cs / cs.shift(1))
        pre_sigma[s] = (0.65 * lr.rolling(20, min_periods=10).std() + 0.35 * (atr(daily_bars[s], 14) / cs) / 1.25).clip(0.004, 0.12)

    soxx_votes = ensemble_votes(daily_bars["SOXX"]) if "SOXX" in daily_bars and len(daily_bars["SOXX"]) >= 60 else pd.Series(dtype=int)

    dates = close_panel.index
    start_i = max(160, len(dates) - p.eval_days - 1)
    if start_i >= len(dates) - 5:
        return {"error": "日线历史长度不足"}

    stat_models: dict[str, tuple[np.ndarray, np.ndarray, float, np.ndarray]] = {}
    drag = p.fill_drag
    fee = p.fee_per_order

    cash = p.capital
    pos_map: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    eq_rows: list[tuple[pd.Timestamp, float]] = []

    for step_i, di in enumerate(range(start_i, len(dates) - 1)):
        dt = dates[di]
        sub_p = close_panel.iloc[: di + 1]
        retrain = (step_i % max(1, p.retrain_every) == 0)
        sv = int(soxx_votes.loc[dt]) if dt in soxx_votes.index else 0

        day_sig: dict[str, tuple[float, float, float, float]] = {}
        for sym in syms:
            df = daily_bars[sym]
            if dt not in df.index:
                continue
            pos = int(df.index.get_loc(dt))
            if pos < 100 or pos + 1 >= len(df):
                continue
            cur_c = float(df["close"].iloc[pos])
            sig1 = float(pre_sigma[sym].iloc[pos]) if pd.notna(pre_sigma[sym].iloc[pos]) else 0.025

            peer = estimate_peer_layer(
                sym, groups.get(sym, "default"), groups, sub_p, cur_c,
                live_prices=None, window=60, min_obs=35, horizon=5, reversion_frac=0.50, ridge_alpha=2.0,
            )
            stat_fv: float | None = None
            xf = pre_feats[sym].iloc[pos].to_numpy(dtype=float)
            if not np.isnan(xf).any():
                if retrain or sym not in stat_models:
                    te = pos - 3
                    ts = max(0, te - 251)
                    if te - ts + 1 >= 80:
                        ca = df["close"].to_numpy(dtype=float)
                        Xsl = pre_feats[sym].iloc[ts : te + 1].to_numpy(dtype=float)
                        ysl = np.log(ca[ts + 3 : te + 4] / ca[ts : te + 1])
                        v = (~np.isnan(Xsl).any(axis=1)) & np.isfinite(ysl)
                        if int(v.sum()) >= 80:
                            Xt, yt = Xsl[v], ysl[v]
                            mx = Xt.mean(axis=0)
                            sx = np.where(Xt.std(axis=0) < 1e-8, 1.0, Xt.std(axis=0))
                            Xs = (Xt - mx) / sx
                            my = float(yt.mean())
                            w = np.linalg.solve(Xs.T @ Xs + 80.0 * np.eye(Xs.shape[1]), Xs.T @ (yt - my))
                            stat_models[sym] = (mx, sx, my, w)
                if sym in stat_models:
                    mx, sx, my, w = stat_models[sym]
                    pr = float(np.clip(my + ((xf - mx) / sx) @ w, -1.8 * sig1 * np.sqrt(3.0), 1.8 * sig1 * np.sqrt(3.0)))
                    stat_fv = float(cur_c * np.exp(pr))

            parts: list[tuple[float, float]] = []
            if peer.get("peer_fv"):
                parts.append((float(peer["peer_fv"]), 0.35 * float(np.clip((peer.get("peer_r2") or 0.0) / 0.60, 0.15, 1.0))))
            if stat_fv:
                parts.append((stat_fv, 0.50))
            if not parts:
                continue
            tw = sum(w for _, w in parts)
            fv = float(np.exp(sum(w * np.log(v) for v, w in parts) / tw))
            z = float(np.log(cur_c / fv) / sig1)
            nxt_o = float(df["open"].iloc[pos + 1])
            day_sig[sym] = (cur_c, fv, z, nxt_o)

        # 1) 卖出检查（高卖止盈 / 止损 / 超时 / 熊市趋势退出）
        for sym in list(pos_map.keys()):
            if sym not in day_sig:
                continue
            pos_d = pos_map[sym]
            cur_c, fv, z, nxt_o = day_sig[sym]
            pos_d["days"] += 1
            ret_now = cur_c / pos_d["entry"] - 1.0
            reason = None
            if ret_now <= -p.stop_pct:
                reason = "stop"
            elif sv <= p.require_soxx_above:
                reason = "bear_regime"
            elif z >= p.exit_z:
                reason = "revert_fv"
            elif pos_d["days"] >= p.max_hold_days:
                reason = "timeout"
            if reason:
                fill = nxt_o * (1.0 - drag)
                pnl = pos_d["qty"] * (fill - pos_d["entry"]) - 2.0 * fee
                cash += pos_d["qty"] * fill - fee
                trades.append(
                    {
                        "symbol": sym,
                        "group": pos_d["group"],
                        "entry_date": pos_d["entry_date"],
                        "exit_date": str(dates[di + 1].date()),
                        "qty": pos_d["qty"],
                        "entry_price": round(pos_d["entry"], 2),
                        "exit_price": round(fill, 2),
                        "hold_days": pos_d["days"],
                        "pnl": round(pnl, 2),
                        "ret_pct": round((fill / pos_d["entry"] - 1.0) * 100.0, 2),
                        "reason": reason,
                    }
                )
                pos_map.pop(sym)

        # 2) 低买检查（非熊市趋势且未满仓）
        if sv > p.require_soxx_above and len(pos_map) < p.max_positions:
            cands = [
                (sym, z, nxt_o, fv)
                for sym, (_, fv, z, nxt_o) in day_sig.items()
                if sym not in pos_map and z <= p.entry_z
            ]
            cands.sort(key=lambda x: x[1])
            for sym, z, nxt_o, fv in cands:
                if len(pos_map) >= p.max_positions:
                    break
                grp = groups.get(sym, "default")
                if sum(1 for v in pos_map.values() if v["group"] == grp) >= p.max_per_group:
                    continue
                fill = nxt_o * (1.0 + drag)
                qty = int(min(p.notional_per_trade, cash - fee) // fill)
                if qty <= 0:
                    continue
                cash -= qty * fill + fee
                pos_map[sym] = {
                    "group": grp,
                    "qty": qty,
                    "entry": fill,
                    "entry_date": str(dates[di + 1].date()),
                    "entry_fv": fv,
                    "days": 0,
                }

        eq = cash + sum(v["qty"] * day_sig[s][0] for s, v in pos_map.items() if s in day_sig)
        eq_rows.append((dt, eq))

    if not eq_rows:
        return {"error": "回测未生成权益曲线"}

    eq_s = pd.Series([e for _, e in eq_rows], index=[d for d, _ in eq_rows], dtype=float)
    rets = eq_s.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(252.0)) if len(rets) > 2 and rets.std(ddof=1) > 1e-12 else 0.0
    max_dd = float((eq_s / eq_s.cummax() - 1.0).min() * 100.0)
    mid = len(eq_s) // 2
    y1 = float((eq_s.iloc[mid] / p.capital - 1.0) * 100.0)
    y2 = float((eq_s.iloc[-1] / eq_s.iloc[mid] - 1.0) * 100.0)
    last40 = float((eq_s.iloc[-1] / eq_s.iloc[max(0, len(eq_s) - 40)] - 1.0) * 100.0)

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw, gl = sum(t["pnl"] for t in wins), abs(sum(t["pnl"] for t in losses))
    pf = round(gw / gl, 2) if gl > 0 else (99.0 if gw > 0 else 0.0)

    by_reason: dict[str, dict[str, Any]] = {}
    for t in trades:
        b = by_reason.setdefault(t["reason"], {"count": 0, "pnl": 0.0})
        b["count"] += 1
        b["pnl"] = round(b["pnl"] + t["pnl"], 2)

    by_group: dict[str, dict[str, Any]] = {}
    for t in trades:
        b = by_group.setdefault(t["group"], {"count": 0, "wins": 0, "pnl": 0.0})
        b["count"] += 1
        b["wins"] += int(t["pnl"] > 0)
        b["pnl"] = round(b["pnl"] + t["pnl"], 2)
    for g in by_group.values():
        g["win_rate_pct"] = round(g["wins"] / g["count"] * 100.0, 1) if g["count"] else 0.0

    return {
        "period": f"{eq_s.index[0].date()} → {eq_s.index[-1].date()}",
        "days": len(eq_s),
        "initial_capital": p.capital,
        "final_equity": round(float(eq_s.iloc[-1]), 2),
        "total_pnl": round(float(eq_s.iloc[-1] - p.capital), 2),
        "total_return_pct": round((float(eq_s.iloc[-1]) / p.capital - 1.0) * 100.0, 2),
        "first_half_return_pct": round(y1, 2),
        "second_half_return_pct": round(y2, 2),
        "last_40d_return_pct": round(last40, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100.0, 1) if trades else 0.0,
        "profit_factor": pf,
        "avg_pnl_per_trade": round(sum(t["pnl"] for t in trades) / len(trades), 2) if trades else 0.0,
        "by_reason": by_reason,
        "by_group": by_group,
        "recent_trades": trades[-10:],
    }
