"""三层每日公允价值模型（Pre-Market Daily Fair Value System）与滚动样本外回测。

三层结构：
  1. 相对定价层（Peer StatArb）：用过去 W 天对大盘（SPY/QQQ）+ 同板块篮子的滚动 Ridge 回归，
     计算当日“按历史联动逻辑应该在什么价位”（支持代入盘前价实时映射）。
  2. 短期量价与波动均衡层（Stat/Vol）：用严格无未来函数的滚动 Ridge 回归预测短期预期收益，
     结合 20 日收盘波动率 + ATR(14) 给出当日 1σ（68%）与 90% 合理价格区间。
  3. 基本面锚定与事件保护层（Fundamental & Regime）：用同板块增长调整 Forward P/E（或 P/S）
     与折现后的分析师目标价中位数构建中期基本面锚，以有界“引力拉回”参与当日战术公允价，
     并联动宏观事件与财报窗口（财报前后自动降权历史板块回归，避免把业绩重估误判为错杀）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .indicators import atr, rsi

DEFAULT_CFG: dict[str, Any] = {
    "peer_window": 60,
    "peer_min_obs": 35,
    "peer_horizon": 5,
    "peer_reversion_frac": 0.50,
    "peer_ridge_alpha": 2.0,
    "stat_train_window": 252,
    "stat_min_obs": 80,
    "stat_horizon": 3,
    "stat_ridge_alpha": 80.0,
    "fund_pull_strength": 0.12,
    "fund_discount_rate": 0.08,
    "weights": {"peer": 0.35, "stat": 0.50, "fund": 0.15},
    "earnings_window_days": 3,
    "backtest_eval_days": 500,
    "backtest_retrain_every": 5,
}


def merge_cfg(user_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {**DEFAULT_CFG, "weights": dict(DEFAULT_CFG["weights"])}
    if user_cfg:
        for k, v in user_cfg.items():
            if k == "weights" and isinstance(v, dict):
                cfg["weights"].update(v)
            else:
                cfg[k] = v
    return cfg


@dataclass
class StockFairValue:
    symbol: str
    group: str
    as_of_bar_date: str
    ref_price: float
    last_close: float
    price_source: str
    overnight_chg_pct: float     # (ref_price / last_close - 1) * 100；相对昨收的盘后/夜盘/盘前涨跌
    fair_value: float
    mispricing_pct: float        # (ref_price / fair_value - 1) * 100；负值=偏低/低估，正值=偏高/高估
    z_score: float               # ln(ref_price / fair_value) / sigma_1d
    sigma_1d_pct: float          # 当日预期 1σ 波动率（%）
    low_1s: float
    high_1s: float
    low_90: float
    high_90: float
    signal: str                  # 明显低估 / 小幅低估 / 合理区间 / 小幅高估 / 明显高估
    # 第一层：同板块相对定价
    peer_fv: float | None = None
    peer_mispricing_pct: float | None = None
    peer_z: float | None = None
    peer_r2: float | None = None
    peer_half_life: float | None = None
    peer_names: list[str] = field(default_factory=list)
    # 第二层：短期统计量价
    stat_fv: float | None = None
    stat_pred_ret_pct: float | None = None
    # 第三层：基本面锚定
    fund_fv: float | None = None          # 中期基本面合理价（12m 折现 + 同行倍数融合）
    fund_tactical_fv: float | None = None # 经引力缩放后参与当日定价的目标价
    fund_gap_pct: float | None = None     # (fund_fv / ref_price - 1) * 100；正值=基本面有上行空间
    forward_pe: float | None = None
    peer_median_pe: float | None = None
    target_median: float | None = None
    analyst_count: int = 0
    # 融合权重与风控提示
    weights: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    attribution: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------- 基础数值工具 ----------------

def _ridge_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """标准化后的 Ridge 回归，返回 (y_hat_train, y_hat_eval, r2_train)。"""
    mu_x = X_train.mean(axis=0)
    sd_x = X_train.std(axis=0)
    sd_x = np.where(sd_x < 1e-8, 1.0, sd_x)

    Xs = (X_train - mu_x) / sd_x
    mu_y = float(y_train.mean())
    yc = y_train - mu_y

    k = Xs.shape[1]
    A = Xs.T @ Xs + alpha * np.eye(k)
    b = Xs.T @ yc
    w = np.linalg.solve(A, b)

    y_hat_train = mu_y + Xs @ w
    Xe = (X_eval - mu_x) / sd_x
    y_hat_eval = mu_y + Xe @ w

    ss_res = float(np.sum((y_train - y_hat_train) ** 2))
    ss_tot = float(np.sum((y_train - mu_y) ** 2))
    r2 = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0)) if ss_tot > 1e-12 else 0.0
    return y_hat_train, y_hat_eval, r2


def _ou_half_life(residuals: np.ndarray) -> float | None:
    """估计残差序列的 Ornstein-Uhlenbeck 均值回归半衰期（交易日）。"""
    if len(residuals) < 15:
        return None
    x = residuals[:-1]
    dy = np.diff(residuals)
    x_c = x - x.mean()
    denom = float(np.dot(x_c, x_c))
    if denom < 1e-12:
        return None
    phi = float(np.dot(x_c, dy - dy.mean()) / denom)
    if phi >= -1e-4 or phi <= -1.99:
        return None
    hl = -np.log(2.0) / np.log(1.0 + max(phi, -0.999))
    if not np.isfinite(hl) or hl <= 0:
        return None
    return float(np.clip(hl, 1.0, 120.0))


def _signal_label(z_score: float) -> str:
    if z_score <= -0.90:
        return "明显低估"
    if z_score <= -0.35:
        return "小幅低估"
    if z_score >= 0.90:
        return "明显高估"
    if z_score >= 0.35:
        return "小幅高估"
    return "合理区间"


def estimate_daily_sigma(df: pd.DataFrame) -> float:
    """混合 20 日收盘对数收益波动率与换算后的 ATR(14)/Close，校准当日 1σ（68.3%）波动率。"""
    if len(df) < 16:
        return 0.02
    close = df["close"].astype(float)
    log_ret = np.log(close / close.shift(1)).dropna()
    vol_20 = float(log_ret.tail(20).std(ddof=1)) if len(log_ret) >= 5 else 0.02
    atr_series = atr(df, 14).dropna()
    # ATR 为日内高低波幅，除以 1.25 折算为等价收盘对收盘 1σ
    atr_sigma = float((atr_series.iloc[-1] / close.iloc[-1]) / 1.25) if len(atr_series) and close.iloc[-1] > 0 else vol_20
    if not np.isfinite(vol_20):
        vol_20 = 0.02
    if not np.isfinite(atr_sigma):
        atr_sigma = vol_20
    return float(np.clip(0.65 * vol_20 + 0.35 * atr_sigma, 0.004, 0.12))


# ---------------- 第一层：同板块 + 大盘相对定价（日收益因子回归 + 短期残差锚定） ----------------

def estimate_peer_layer(
    symbol: str,
    group: str,
    groups: dict[str, str],
    close_panel: pd.DataFrame,
    ref_price: float,
    live_prices: dict[str, float] | None = None,
    window: int = 60,
    min_obs: int = 35,
    horizon: int = 5,
    reversion_frac: float = 0.50,
    ridge_alpha: float = 2.0,
) -> dict[str, Any]:
    """用过去 window 天的日对数收益拟合 r_i ~ r_SPY + r_QQQ + r_peer（平稳因子模型），
    再以 horizon 个交易日前为锚，代入最新（或盘前）大盘与同板块累计涨跌算出当日板块联动公允价。
    """
    if symbol not in close_panel.columns:
        return {}
    s_series = close_panel[symbol].dropna()
    need = min_obs + horizon + 2
    if len(s_series) < need:
        return {}

    # 拟合窗口严格早于评估锚点/评估日，杜绝样本内收缩与未来函数
    if live_prices is not None:
        anchor_pos = len(s_series) - horizon
        eval_factor_prices = live_prices
        fit_idx = s_series.index[max(0, len(s_series) - window - 1) : len(s_series)]
    else:
        anchor_pos = len(s_series) - 1 - horizon
        eval_date = s_series.index[-1]
        eval_factor_prices = {
            c: float(close_panel.loc[eval_date, c])
            for c in close_panel.columns
            if pd.notna(close_panel.loc[eval_date, c]) and close_panel.loc[eval_date, c] > 0
        }
        fit_idx = s_series.index[max(0, len(s_series) - window - 2) : len(s_series) - 1]

    if anchor_pos < 0 or len(fit_idx) < min_obs + 1:
        return {}

    anchor_date = s_series.index[anchor_pos]
    anchor_price = float(s_series.iloc[anchor_pos])
    if not np.isfinite(anchor_price) or anchor_price <= 0:
        return {}

    sub = close_panel.reindex(fit_idx)
    y_prices = sub[symbol]
    valid_mask = y_prices.notna() & (y_prices > 0)
    sub = sub.loc[valid_mask]
    if len(sub) < min_obs + 1:
        return {}

    # 候选因子：大盘基准（SPY/QQQ）+ 同板块同行（排除自身）
    def _valid_col(col: str) -> bool:
        return (
            col in sub.columns
            and sub[col].notna().all()
            and (sub[col] > 0).all()
            and pd.notna(close_panel.loc[anchor_date, col])
            and float(close_panel.loc[anchor_date, col]) > 0
            and eval_factor_prices.get(col, 0) > 0
        )

    market_cols = [m for m in ("SPY", "QQQ") if m != symbol and _valid_col(m)]
    raw_peers = [p for p, g in groups.items() if g == group and p != symbol and _valid_col(p)]

    y_ret = np.diff(np.log(sub[symbol].to_numpy(dtype=float)))
    factor_rets: list[np.ndarray] = []
    horizon_factor_moves: list[float] = []

    for m in market_cols:
        factor_rets.append(np.diff(np.log(sub[m].to_numpy(dtype=float))))
        horizon_factor_moves.append(float(np.log(eval_factor_prices[m] / float(close_panel.loc[anchor_date, m]))))

    if raw_peers:
        peer_ret_mat = np.column_stack([np.diff(np.log(sub[p].to_numpy(dtype=float))) for p in raw_peers])
        factor_rets.append(peer_ret_mat.mean(axis=1))
        peer_move = float(
            np.mean([np.log(eval_factor_prices[p] / float(close_panel.loc[anchor_date, p])) for p in raw_peers])
        )
        horizon_factor_moves.append(peer_move)

    if not factor_rets:
        return {}

    X_train = np.column_stack(factor_rets)
    mu_x = X_train.mean(axis=0)
    sd_x = np.where(X_train.std(axis=0) < 1e-8, 1.0, X_train.std(axis=0))
    Xs = (X_train - mu_x) / sd_x
    mu_y = float(y_ret.mean())
    yc = y_ret - mu_y
    w_std = np.linalg.solve(Xs.T @ Xs + ridge_alpha * np.eye(Xs.shape[1]), Xs.T @ yc)
    # 还原到原始收益空间的 beta（不含截距漂移，避免把历史上涨趋势外推成永久偏离）
    betas = w_std / sd_x

    y_hat_train = mu_y + Xs @ w_std
    residuals = y_ret - y_hat_train
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_ret - mu_y) ** 2))
    r2 = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0)) if ss_tot > 1e-12 else 0.0

    # 计算短期累计残差的半衰期（对累计残差序列做 OU 估计）
    cum_res = np.cumsum(residuals)
    half_life = _ou_half_life(cum_res)
    res_std_1d = max(float(np.std(residuals, ddof=1)), 0.004)
    horizon_res_std = res_std_1d * np.sqrt(max(1, horizon))

    # 过去 horizon 天按因子联动“应该走出的累计对数收益”
    implied_cum_ret = float(np.dot(betas, np.array(horizon_factor_moves, dtype=float)))
    raw_implied_log_p = float(np.log(anchor_price) + implied_cum_ret)
    log_ref = float(np.log(ref_price))
    raw_dislocation = log_ref - raw_implied_log_p

    # 短期回归阻尼：通常 1~3 天只会回补一部分偏离，避免把全部残差当作次日目标
    damped_log_fv = log_ref - float(np.clip(raw_dislocation, -0.20, 0.20)) * reversion_frac
    peer_fv = float(np.exp(damped_log_fv))
    peer_mispricing_pct = (ref_price / peer_fv - 1.0) * 100.0
    peer_z = float(raw_dislocation / horizon_res_std)

    return {
        "peer_fv": peer_fv,
        "peer_raw_fv": float(np.exp(np.clip(raw_implied_log_p, log_ref - 0.25, log_ref + 0.25))),
        "peer_mispricing_pct": peer_mispricing_pct,
        "peer_z": peer_z,
        "peer_r2": r2,
        "peer_res_std": horizon_res_std,
        "peer_half_life": half_life,
        "peer_names": raw_peers if raw_peers else market_cols,
    }


# ---------------- 第二层：短期统计量价模型 ----------------

def build_stat_features(df: pd.DataFrame, bench_close: pd.Series | None = None) -> pd.DataFrame:
    """构建严格无未来函数的日线量价特征表（全部基于当日收盘及之前数据）。"""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float).fillna(0.0)

    log_c = np.log(close.clip(lower=1e-6))
    ret_1d = log_c.diff(1)
    ret_3d = log_c.diff(3)
    ret_5d = log_c.diff(5)
    ret_20d = log_c.diff(20)

    vol_20 = ret_1d.rolling(20, min_periods=10).std().clip(lower=0.004)
    atr_14 = atr(df, 14).clip(lower=close * 0.003)

    # 20 日成交量加权均价（VWAP 代理）
    pv = (close * vol).rolling(20, min_periods=10).sum()
    vv = vol.rolling(20, min_periods=10).sum()
    vwap_20 = np.where(vv > 0, pv / vv, close.rolling(20, min_periods=10).mean())
    vwap_20 = pd.Series(vwap_20, index=df.index)

    ema_50 = close.ewm(span=50, adjust=False).mean()
    rsi_14 = rsi(close, 14)

    hl_span = (high - low).replace(0.0, np.nan)
    clv = ((2.0 * close - high - low) / hl_span).fillna(0.0)

    vol_ma20 = vol.rolling(20, min_periods=10).mean()
    vol_ratio = np.log((vol + 1.0) / (vol_ma20 + 1.0)).clip(-2.0, 2.0)
    signed_vol_shock = np.sign(ret_1d.fillna(0.0)) * vol_ratio

    feats = pd.DataFrame(
        {
            "ret_1d_z": ret_1d / vol_20,
            "ret_3d_z": ret_3d / (vol_20 * np.sqrt(3.0)),
            "ret_5d_z": ret_5d / (vol_20 * np.sqrt(5.0)),
            "dist_vwap20": (close - vwap_20) / atr_14,
            "dist_ema50": (close - ema_50) / atr_14,
            "rsi14_c": (rsi_14 - 50.0) / 25.0,
            "clv": clv,
            "signed_vol_shock": signed_vol_shock,
        },
        index=df.index,
    )

    if bench_close is not None and len(bench_close) > 20:
        b_aligned = bench_close.reindex(df.index).ffill()
        b_log = np.log(b_aligned.clip(lower=1e-6))
        feats["rel_bench_5d"] = (ret_5d - b_log.diff(5)) / (vol_20 * np.sqrt(5.0))
        feats["rel_bench_20d"] = (ret_20d - b_log.diff(20)) / (vol_20 * np.sqrt(20.0))

    return feats.replace([np.inf, -np.inf], np.nan).clip(-4.0, 4.0)


def estimate_stat_layer(
    df: pd.DataFrame,
    bench_close: pd.Series | None = None,
    horizon: int = 3,
    train_window: int = 252,
    min_obs: int = 80,
    ridge_alpha: float = 80.0,
    feats_precomputed: pd.DataFrame | None = None,
    eval_pos: int | None = None,
) -> dict[str, Any]:
    """用严格因果的滚动 Ridge 回归预测未来 horizon 天的预期收益，给出统计均衡公允价。"""
    feats = feats_precomputed if feats_precomputed is not None else build_stat_features(df, bench_close)
    pos = (len(df) - 1) if eval_pos is None else eval_pos
    if pos < min_obs + horizon:
        return {}

    close = df["close"].astype(float).to_numpy()
    last_close = float(close[pos])
    if not np.isfinite(last_close) or last_close <= 0:
        return {}

    # 评估行特征（在 pos 收盘时完全已知）
    x_eval_row = feats.iloc[pos].to_numpy(dtype=float)
    if np.isnan(x_eval_row).any():
        return {}

    # 训练样本截止到 pos - horizon，保证每个标签 close[k + horizon] / close[k] 在 pos 当日已完全兑现
    train_end = pos - horizon
    train_start = max(0, train_end - train_window + 1)
    if train_end - train_start + 1 < min_obs:
        return {}

    X_slice = feats.iloc[train_start : train_end + 1].to_numpy(dtype=float)
    y_slice = np.log(close[train_start + horizon : train_end + horizon + 1] / close[train_start : train_end + 1])

    valid = (~np.isnan(X_slice).any(axis=1)) & np.isfinite(y_slice)
    if int(valid.sum()) < min_obs:
        return {}

    X_train = X_slice[valid]
    y_train = y_slice[valid]

    _, y_hat_eval, r2 = _ridge_fit_predict(X_train, y_train, x_eval_row.reshape(1, -1), alpha=ridge_alpha)
    sigma_1d = estimate_daily_sigma(df.iloc[: pos + 1])
    # 预测收益裁剪在 ±1.8 * sigma_1d * sqrt(horizon) 之内，避免极端样本外推
    cap = 1.8 * sigma_1d * np.sqrt(max(1, horizon))
    pred_log_ret = float(np.clip(y_hat_eval[0], -cap, cap))
    stat_fv = float(last_close * np.exp(pred_log_ret))

    return {
        "stat_fv": stat_fv,
        "stat_pred_ret_pct": float((np.exp(pred_log_ret) - 1.0) * 100.0),
        "stat_r2": r2,
        "sigma_1d": sigma_1d,
    }


# ---------------- 第三层：基本面锚定与事件保护 ----------------

def _valid_median(values: list[float | None], lo: float, hi: float) -> float | None:
    clean = [v for v in values if v is not None and np.isfinite(v) and lo < v < hi]
    return float(np.median(clean)) if clean else None


def compute_group_fundamental_medians(
    groups: dict[str, str],
    fundamentals: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, float | None]], dict[str, float | None]]:
    """按板块汇总 Forward P/E、P/S、营收增速中位数；同时计算全市场个股保底中位数。"""
    by_group: dict[str, list[dict[str, Any]]] = {}
    all_eq: list[dict[str, Any]] = []
    for sym, fd in fundamentals.items():
        if fd.get("quote_type") == "ETF" or groups.get(sym) == "index":
            continue
        all_eq.append(fd)
        by_group.setdefault(groups.get(sym, "default"), []).append(fd)

    def _agg(rows: list[dict[str, Any]]) -> dict[str, float | None]:
        return {
            "forward_pe": _valid_median([r.get("forward_pe") for r in rows], 2.0, 150.0),
            "price_to_sales": _valid_median([r.get("price_to_sales") for r in rows], 0.1, 80.0),
            "revenue_growth": _valid_median([r.get("revenue_growth") for r in rows], -0.6, 2.5),
        }

    global_med = _agg(all_eq)
    group_med: dict[str, dict[str, float | None]] = {}
    for g, rows in by_group.items():
        m = _agg(rows)
        # 板块内有效样本不足 2 只时回退全局个股中位数
        for key in ("forward_pe", "price_to_sales", "revenue_growth"):
            valid_cnt = sum(1 for r in rows if r.get(key) is not None and np.isfinite(r.get(key)))
            if valid_cnt < 2 or m[key] is None:
                m[key] = global_med[key]
        group_med[g] = m
    return group_med, global_med


def estimate_fundamental_layer(
    symbol: str,
    group: str,
    ref_price: float,
    fd: dict[str, Any] | None,
    group_medians: dict[str, dict[str, float | None]],
    global_medians: dict[str, float | None],
    pull_strength: float = 0.12,
    discount_rate: float = 0.08,
) -> dict[str, Any]:
    """构建基本面锚（同板块增长调整估值倍数 + 折现分析师一致目标价）与当日战术引力价。"""
    if not fd or fd.get("quote_type") == "ETF" or group == "index":
        return {}

    med = group_medians.get(group) or global_medians
    peer_pe = med.get("forward_pe")
    peer_ps = med.get("price_to_sales")
    peer_rev_g = med.get("revenue_growth")

    fwd_eps = fd.get("forward_eps")
    fwd_pe = fd.get("forward_pe")
    ps = fd.get("price_to_sales")
    rev_g = fd.get("revenue_growth")

    # 1) 同板块增长调整估值倍数公允价
    mult_fv: float | None = None
    if fwd_eps and fwd_eps > 0 and peer_pe and peer_pe > 0:
        growth_diff = 0.0
        if rev_g is not None and peer_rev_g is not None:
            growth_diff = float(np.clip(rev_g - peer_rev_g, -0.6, 0.8))
        adj = float(np.clip(1.0 + 0.5 * growth_diff, 0.70, 1.40))
        mult_fv = float(fwd_eps * peer_pe * adj)
    elif ps and ps > 0 and peer_ps and peer_ps > 0 and ref_price > 0:
        sales_per_share = ref_price / ps
        mult_fv = float(sales_per_share * peer_ps)

    # 2) 分析师一致目标价（中位数优先，折现 12 个月预期乐观偏差/资金成本）
    target_med = fd.get("target_median") or fd.get("target_mean")
    analyst_cnt = int(fd.get("analyst_count") or 0)
    analyst_fv: float | None = None
    if target_med and target_med > 0:
        analyst_fv = float(target_med / (1.0 + discount_rate))

    # 融合两个基本面锚
    anchors: list[tuple[float, float]] = []
    if mult_fv and mult_fv > 0:
        anchors.append((mult_fv, 0.55))
    if analyst_fv and analyst_fv > 0:
        w_a = 0.45 if analyst_cnt >= 5 else 0.30
        anchors.append((analyst_fv, w_a))

    if not anchors:
        return {
            "forward_pe": fwd_pe,
            "peer_median_pe": peer_pe,
            "target_median": target_med,
            "analyst_count": analyst_cnt,
        }

    total_w = sum(w for _, w in anchors)
    raw_fund_fv = float(np.exp(sum(w * np.log(v) for v, w in anchors) / total_w))
    # 防止极端异常财报数据导致失真
    fund_fv = float(np.clip(raw_fund_fv, ref_price * 0.40, ref_price * 2.20))
    fund_gap_pct = (fund_fv / ref_price - 1.0) * 100.0

    # 有界基本面引力：中期估值差不会在一天内完全兑现，按 pull_strength 温和牵引当日公允价
    log_gap = float(np.clip(np.log(fund_fv / ref_price), -0.35, 0.35))
    fund_tactical_fv = float(ref_price * np.exp(log_gap * pull_strength))

    return {
        "fund_fv": fund_fv,
        "fund_tactical_fv": fund_tactical_fv,
        "fund_gap_pct": fund_gap_pct,
        "forward_pe": fwd_pe,
        "peer_median_pe": peer_pe,
        "target_median": target_med,
        "analyst_count": analyst_cnt,
    }


# ---------------- 综合融合引擎 ----------------

def _build_attribution(
    symbol: str,
    peer: dict[str, Any],
    stat: dict[str, Any],
    fund: dict[str, Any],
    flags: list[str],
    signal: str,
) -> str:
    parts: list[str] = []
    if peer.get("peer_mispricing_pct") is not None:
        pm = peer["peer_mispricing_pct"]
        peers_str = "/".join(peer.get("peer_names", [])[:3])
        if pm <= -1.0:
            parts.append(f"落后 {peers_str} 联动 {abs(pm):.1f}%")
        elif pm >= 1.0:
            parts.append(f"跑赢 {peers_str} 联动 +{pm:.1f}%")
    if stat.get("stat_pred_ret_pct") is not None:
        sr = stat["stat_pred_ret_pct"]
        if sr >= 0.6:
            parts.append(f"量价模型偏多(+{sr:.1f}%)")
        elif sr <= -0.6:
            parts.append(f"量价模型偏空({sr:.1f}%)")
    if fund.get("fund_gap_pct") is not None:
        fg = fund["fund_gap_pct"]
        if fg >= 10.0:
            parts.append(f"基本面锚偏高(+{fg:.0f}%)")
        elif fg <= -10.0:
            parts.append(f"基本面锚偏低({fg:.0f}%)")
    if "POST_EARNINGS_REPRICING" in flags:
        parts.append("财报刚落地(已降权板块回归)")
    elif "EARNINGS_SOON" in flags:
        parts.append("临近财报(波动放大)")
    if "KOSPI_SHOCK" in flags:
        parts.append("KOSPI急跌冲击(已停用基本面看多拉回)")
    elif "BEAR_REGIME" in flags:
        parts.append("下跌趋势(已降权看多基本面锚)")
    if not parts:
        return f"三层信号均衡（{signal}）"
    return "；".join(parts)


SEMI_SENSITIVE_GROUPS = {"semis", "memory", "optical"}
SEMI_SENSITIVE_SYMS = {"NVDA", "SOXL", "SOXS"}


def _rolling_beta(stock_close: pd.Series, bench_close: pd.Series, window: int = 60) -> float:
    """计算过去 window 个交易日相对基准指数的滚动 Beta，用于夜盘/盘前大盘跳空同步重估。"""
    joined = pd.concat([stock_close.rename("s"), bench_close.rename("b")], axis=1).dropna().tail(window + 1)
    if len(joined) < 20:
        return 1.0
    sr = np.diff(np.log(joined["s"].to_numpy(dtype=float)))
    br = np.diff(np.log(joined["b"].to_numpy(dtype=float)))
    var_b = float(np.var(br, ddof=1))
    if var_b < 1e-10:
        return 1.0
    beta = float(np.cov(sr, br, ddof=1)[0, 1] / var_b)
    if not np.isfinite(beta):
        return 1.0
    return float(np.clip(beta, 0.3, 2.5))


def compute_daily_fair_values(
    bars: dict[str, pd.DataFrame],
    groups: dict[str, str],
    symbols: list[str] | None = None,
    fundamentals: dict[str, dict[str, Any]] | None = None,
    event_context: dict[str, Any] | None = None,
    live_quotes: dict[str, dict[str, Any]] | None = None,
    regime_context: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[StockFairValue]:
    """对整个标的池运行三层估值融合，生成开盘前/夜盘实时公允价值列表。"""
    c = merge_cfg(cfg)
    fundamentals = fundamentals or {}
    event_context = event_context or {"macro_alerts": [], "earnings_flags": {}}
    live_quotes = live_quotes or {}
    regime_context = regime_context or {}
    target_syms = [s.upper() for s in (symbols or list(groups))]

    # 拼装收盘价矩阵
    close_cols = {s: df["close"].astype(float) for s, df in bars.items() if len(df) > 0}
    if not close_cols:
        return []
    close_panel = pd.DataFrame(close_cols).sort_index()
    bench_close = close_panel["QQQ"] if "QQQ" in close_panel.columns else close_panel.get("SPY")

    # 组装最新参考价：优先级 Webull 24h 实时价(含夜盘OVN) > Yahoo 盘前/盘后/现价 > 日线最新收盘
    live_prices: dict[str, float] = {}
    price_sources: dict[str, str] = {}
    for s in close_panel.columns:
        last_close = float(close_panel[s].dropna().iloc[-1])
        lq = live_quotes.get(s) or {}
        fd = fundamentals.get(s) or {}
        lp = lq.get("price")
        rp = fd.get("ref_price")
        if lp and np.isfinite(lp) and lp > 0:
            live_prices[s] = float(lp)
            price_sources[s] = str(lq.get("source") or "overnight")
        elif rp and np.isfinite(rp) and rp > 0:
            live_prices[s] = float(rp)
            price_sources[s] = str(fd.get("ref_source") or "regular")
        else:
            live_prices[s] = last_close
            price_sources[s] = "close"

    # 计算基准指数的夜盘/盘前累计跳空，用于同步平移第 2 层统计均衡中枢
    bench_ovn_log: dict[str, float] = {}
    for b_sym in ("QQQ", "SPY"):
        if b_sym in close_panel.columns:
            b_last = float(close_panel[b_sym].dropna().iloc[-1])
            b_live = live_prices.get(b_sym, b_last)
            if b_last > 0 and b_live > 0:
                bench_ovn_log[b_sym] = float(np.log(b_live / b_last))

    group_medians, global_medians = compute_group_fundamental_medians(groups, fundamentals)
    macro_alerts = event_context.get("macro_alerts") or []
    earnings_flags = event_context.get("earnings_flags") or {}
    soxx_regime = regime_context.get("soxx_regime")
    qqq_regime = regime_context.get("qqq_regime")
    kospi_move_pct = regime_context.get("kospi_move_pct")

    results: list[StockFairValue] = []
    for sym in target_syms:
        if sym not in bars or len(bars[sym]) < 20:
            continue
        df = bars[sym]
        grp = groups.get(sym, "default")
        last_close = float(df["close"].iloc[-1])
        bar_date = str(df.index[-1].date())
        ref_price = live_prices.get(sym, last_close)
        p_src = price_sources.get(sym, "close")
        ovn_chg_pct = (ref_price / last_close - 1.0) * 100.0 if last_close > 0 else 0.0

        flags: list[str] = []
        if macro_alerts:
            flags.append("MACRO_EVENT")
        ef = earnings_flags.get(sym)
        if ef:
            flags.append(ef["status"])

        is_semi = (grp in SEMI_SENSITIVE_GROUPS) or (sym in SEMI_SENSITIVE_SYMS)
        in_bear = (is_semi and soxx_regime == -1) or (grp != "defensive" and qqq_regime == -1)
        in_kospi_shock = is_semi and (kospi_move_pct is not None and kospi_move_pct <= -1.5)
        if in_bear:
            flags.append("BEAR_REGIME")
        if in_kospi_shock:
            flags.append("KOSPI_SHOCK")

        # Layer 1: Peer StatArb（代入实时夜盘/盘前大盘与同行价）
        peer = estimate_peer_layer(
            symbol=sym,
            group=grp,
            groups=groups,
            close_panel=close_panel,
            ref_price=ref_price,
            live_prices=live_prices,
            window=int(c["peer_window"]),
            min_obs=int(c["peer_min_obs"]),
            horizon=int(c["peer_horizon"]),
            reversion_frac=float(c["peer_reversion_frac"]),
            ridge_alpha=float(c["peer_ridge_alpha"]),
        )
        if peer and (peer.get("peer_r2") or 0.0) < 0.25:
            flags.append("LOW_PEER_R2")

        # Layer 2: Short-Term Stat/Vol + 夜盘大盘 Beta 同步重估
        stat = estimate_stat_layer(
            df=df,
            bench_close=bench_close,
            horizon=int(c["stat_horizon"]),
            train_window=int(c["stat_train_window"]),
            min_obs=int(c["stat_min_obs"]),
            ridge_alpha=float(c["stat_ridge_alpha"]),
        )
        sigma_1d = float(stat.get("sigma_1d") or estimate_daily_sigma(df))

        stat_fv_val = stat.get("stat_fv")
        if stat_fv_val:
            ref_bench = "SPY" if grp in ("defensive", "index") and sym != "QQQ" else "QQQ"
            if ref_bench not in bench_ovn_log:
                ref_bench = "SPY" if "SPY" in bench_ovn_log else "QQQ"
            b_move = bench_ovn_log.get(ref_bench, 0.0)
            if abs(b_move) > 1e-4 and ref_bench in close_panel.columns:
                beta_b = 1.0 if sym == ref_bench else _rolling_beta(df["close"], close_panel[ref_bench], window=60)
                stat_fv_val = float(stat_fv_val * np.exp(beta_b * b_move))
                stat = {**stat, "stat_fv": stat_fv_val}

        # Layer 3: Fundamental Anchor
        fund = estimate_fundamental_layer(
            symbol=sym,
            group=grp,
            ref_price=ref_price,
            fd=fundamentals.get(sym),
            group_medians=group_medians,
            global_medians=global_medians,
            pull_strength=float(c["fund_pull_strength"]),
            discount_rate=float(c["fund_discount_rate"]),
        )

        # 动态权重融合（含趋势与 KOSPI 冲击保护）
        bw = c["weights"]
        raw_w: dict[str, float] = {}
        layer_fvs: dict[str, float] = {}

        if peer.get("peer_fv"):
            r2_scale = float(np.clip((peer.get("peer_r2") or 0.0) / 0.60, 0.15, 1.0))
            w_p = float(bw["peer"]) * r2_scale
            if ef:  # 财报前后窗口：历史联动失效概率高，自动降权
                w_p *= 0.40
            raw_w["peer"] = w_p
            layer_fvs["peer"] = float(peer["peer_fv"])

        if stat.get("stat_fv"):
            raw_w["stat"] = float(bw["stat"])
            layer_fvs["stat"] = float(stat["stat_fv"])

        if fund.get("fund_tactical_fv"):
            w_f = float(bw["fund"])
            if ef and ef.get("status") == "POST_EARNINGS_REPRICING":
                w_f = max(w_f, 0.25)
            # 熊市/亚太半导体冲击时，滞后的卖方目标价不应强行把当日公允价往上拉
            if float(fund["fund_tactical_fv"]) > ref_price:
                if in_kospi_shock:
                    w_f = 0.0
                elif in_bear:
                    w_f *= 0.25
            if w_f > 0:
                raw_w["fund"] = w_f
                layer_fvs["fund"] = float(fund["fund_tactical_fv"])

        if layer_fvs:
            tot = sum(raw_w.values())
            norm_w = {k: round(v / tot, 3) for k, v in raw_w.items()}
            composite_fv = float(np.exp(sum((raw_w[k] / tot) * np.log(layer_fvs[k]) for k in layer_fvs)))
        else:
            norm_w = {}
            composite_fv = ref_price

        mispricing_pct = (ref_price / composite_fv - 1.0) * 100.0
        z_score = float(np.log(ref_price / composite_fv) / sigma_1d)
        sig = _signal_label(z_score)

        low_1s = composite_fv * (1.0 - sigma_1d)
        high_1s = composite_fv * (1.0 + sigma_1d)
        low_90 = composite_fv * (1.0 - 1.645 * sigma_1d)
        high_90 = composite_fv * (1.0 + 1.645 * sigma_1d)

        attr = _build_attribution(sym, peer, stat, fund, flags, sig)
        results.append(
            StockFairValue(
                symbol=sym,
                group=grp,
                as_of_bar_date=bar_date,
                ref_price=round(ref_price, 2),
                last_close=round(last_close, 2),
                price_source=p_src,
                overnight_chg_pct=round(ovn_chg_pct, 2),
                fair_value=round(composite_fv, 2),
                mispricing_pct=round(mispricing_pct, 2),
                z_score=round(z_score, 2),
                sigma_1d_pct=round(sigma_1d * 100.0, 2),
                low_1s=round(low_1s, 2),
                high_1s=round(high_1s, 2),
                low_90=round(low_90, 2),
                high_90=round(high_90, 2),
                signal=sig,
                peer_fv=round(peer["peer_fv"], 2) if peer.get("peer_fv") else None,
                peer_mispricing_pct=round(peer["peer_mispricing_pct"], 2) if peer.get("peer_mispricing_pct") is not None else None,
                peer_z=round(peer["peer_z"], 2) if peer.get("peer_z") is not None else None,
                peer_r2=round(peer["peer_r2"], 3) if peer.get("peer_r2") is not None else None,
                peer_half_life=round(peer["peer_half_life"], 1) if peer.get("peer_half_life") is not None else None,
                peer_names=list(peer.get("peer_names") or []),
                stat_fv=round(stat["stat_fv"], 2) if stat.get("stat_fv") else None,
                stat_pred_ret_pct=round(stat["stat_pred_ret_pct"], 2) if stat.get("stat_pred_ret_pct") is not None else None,
                fund_fv=round(fund["fund_fv"], 2) if fund.get("fund_fv") else None,
                fund_tactical_fv=round(fund["fund_tactical_fv"], 2) if fund.get("fund_tactical_fv") else None,
                fund_gap_pct=round(fund["fund_gap_pct"], 1) if fund.get("fund_gap_pct") is not None else None,
                forward_pe=round(fund["forward_pe"], 1) if fund.get("forward_pe") else None,
                peer_median_pe=round(fund["peer_median_pe"], 1) if fund.get("peer_median_pe") else None,
                target_median=round(fund["target_median"], 2) if fund.get("target_median") else None,
                analyst_count=int(fund.get("analyst_count") or 0),
                weights=norm_w,
                flags=flags,
                attribution=attr,
            )
        )

    # 按偏离度从小到大排（最被低估的排在最前面，方便开盘前一眼扫出机会与风险）
    results.sort(key=lambda r: r.z_score)
    return results


# ---------------- 严格滚动样本外回测（Walk-Forward） ----------------

def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 10:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy(dtype=float)
    ry = pd.Series(y).rank().to_numpy(dtype=float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 10 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def backtest_fair_value(
    bars: dict[str, pd.DataFrame],
    groups: dict[str, str],
    symbols: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """严格无未来函数的 Walk-Forward 回测（验证第 1 层板块联动 + 第 2 层短期统计 + 波动区间校准）。

    说明：免费 Yahoo quoteSummary 不提供历史时点基本面快照，为避免未来函数污染，
    历史回测严格仅验证可回溯的第 1 层与第 2 层以及两者融合结果。
    """
    c = merge_cfg(cfg)
    target_syms = [s.upper() for s in (symbols or list(groups)) if s.upper() in bars]
    close_panel = pd.DataFrame({s: df["close"].astype(float) for s, df in bars.items() if len(df) > 0}).sort_index()
    if close_panel.empty:
        return {"error": "没有可用行情数据"}

    bench_close = close_panel["QQQ"] if "QQQ" in close_panel.columns else close_panel.get("SPY")
    peer_w = int(c["peer_window"])
    peer_min = int(c["peer_min_obs"])
    peer_alpha = float(c["peer_ridge_alpha"])
    stat_w = int(c["stat_train_window"])
    stat_min = int(c["stat_min_obs"])
    stat_h = int(c["stat_horizon"])
    stat_alpha = float(c["stat_ridge_alpha"])
    retrain_every = max(1, int(c["backtest_retrain_every"]))
    eval_days = int(c["backtest_eval_days"])

    # 预计算每只股票的特征和滚动波动率（全部严格因果）
    pre_feats: dict[str, pd.DataFrame] = {}
    pre_sigma: dict[str, pd.Series] = {}
    for sym in target_syms:
        df = bars[sym]
        pre_feats[sym] = build_stat_features(df, bench_close)
        close_s = df["close"].astype(float)
        log_ret = np.log(close_s / close_s.shift(1))
        vol20 = log_ret.rolling(20, min_periods=10).std()
        atr14 = (atr(df, 14) / close_s) / 1.25
        pre_sigma[sym] = (0.65 * vol20 + 0.35 * atr14).clip(0.004, 0.12)

    all_dates = close_panel.index
    warmup = max(peer_w + 5, stat_min + stat_h + 25)
    start_idx = max(warmup, len(all_dates) - eval_days - 5)
    end_idx = len(all_dates) - 6  # 留出 5 天计算前瞻收益
    if end_idx <= start_idx:
        return {"error": f"历史样本不足（需要至少 {warmup + 10} 个交易日）"}

    # 缓存每只股票最近一次训练好的第 2 层 Ridge 参数，每 retrain_every 天重训一次（无前视）
    stat_models: dict[str, tuple[np.ndarray, np.ndarray, float, np.ndarray]] = {}
    records: list[dict[str, Any]] = []

    bw = c["weights"]
    for step_i, dt_idx in enumerate(range(start_idx, end_idx + 1)):
        dt = all_dates[dt_idx]
        sub_panel = close_panel.iloc[: dt_idx + 1]
        do_retrain = (step_i % retrain_every == 0)

        for sym in target_syms:
            df = bars[sym]
            if dt not in df.index:
                continue
            pos = int(df.index.get_loc(dt))
            if pos + 5 >= len(df):
                continue

            cur_close = float(df["close"].iloc[pos])
            if not np.isfinite(cur_close) or cur_close <= 0:
                continue

            sigma_val = pre_sigma[sym].iloc[pos]
            sigma_1d = float(sigma_val) if pd.notna(sigma_val) else 0.02

            # Layer 1: strictly out-of-sample on day dt (fit on past returns, evaluate horizon dislocation at t)
            peer = estimate_peer_layer(
                symbol=sym,
                group=groups.get(sym, "default"),
                groups=groups,
                close_panel=sub_panel,
                ref_price=cur_close,
                live_prices=None,
                window=peer_w,
                min_obs=peer_min,
                horizon=int(c["peer_horizon"]),
                reversion_frac=float(c["peer_reversion_frac"]),
                ridge_alpha=peer_alpha,
            )

            # Layer 2: rolling Ridge (retrained every `retrain_every` days using labels <= pos)
            stat_fv: float | None = None
            feats_df = pre_feats[sym]
            x_eval = feats_df.iloc[pos].to_numpy(dtype=float)
            if not np.isnan(x_eval).any():
                if do_retrain or sym not in stat_models:
                    train_end = pos - stat_h
                    train_start = max(0, train_end - stat_w + 1)
                    if train_end - train_start + 1 >= stat_min:
                        close_arr = df["close"].astype(float).to_numpy()
                        X_sl = feats_df.iloc[train_start : train_end + 1].to_numpy(dtype=float)
                        y_sl = np.log(close_arr[train_start + stat_h : train_end + stat_h + 1] / close_arr[train_start : train_end + 1])
                        valid = (~np.isnan(X_sl).any(axis=1)) & np.isfinite(y_sl)
                        if int(valid.sum()) >= stat_min:
                            Xt = X_sl[valid]
                            yt = y_sl[valid]
                            mu_x = Xt.mean(axis=0)
                            sd_x = np.where(Xt.std(axis=0) < 1e-8, 1.0, Xt.std(axis=0))
                            Xs = (Xt - mu_x) / sd_x
                            mu_y = float(yt.mean())
                            w = np.linalg.solve(Xs.T @ Xs + stat_alpha * np.eye(Xs.shape[1]), Xs.T @ (yt - mu_y))
                            stat_models[sym] = (mu_x, sd_x, mu_y, w)

                if sym in stat_models:
                    mu_x, sd_x, mu_y, w = stat_models[sym]
                    pred_r = float(mu_y + ((x_eval - mu_x) / sd_x) @ w)
                    cap = 1.8 * sigma_1d * np.sqrt(max(1, stat_h))
                    pred_r = float(np.clip(pred_r, -cap, cap))
                    stat_fv = float(cur_close * np.exp(pred_r))

            # Fusion (Layer 1 + Layer 2)
            parts: list[tuple[float, float]] = []
            if peer.get("peer_fv"):
                r2_scale = float(np.clip((peer.get("peer_r2") or 0.0) / 0.60, 0.15, 1.0))
                parts.append((float(peer["peer_fv"]), float(bw["peer"]) * r2_scale))
            if stat_fv:
                parts.append((stat_fv, float(bw["stat"])))
            if not parts:
                continue

            tot_w = sum(w for _, w in parts)
            comp_fv = float(np.exp(sum(w * np.log(fv) for fv, w in parts) / tot_w))

            nxt_1 = float(df["close"].iloc[pos + 1])
            nxt_3 = float(df["close"].iloc[pos + 3])
            nxt_5 = float(df["close"].iloc[pos + 5])

            low_1s = comp_fv * (1.0 - sigma_1d)
            high_1s = comp_fv * (1.0 + sigma_1d)
            low_90 = comp_fv * (1.0 - 1.645 * sigma_1d)
            high_90 = comp_fv * (1.0 + 1.645 * sigma_1d)

            records.append(
                {
                    "date": dt,
                    "symbol": sym,
                    "group": groups.get(sym, "default"),
                    # upside = ln(fv / cur_close)：正值代表模型认为当前被低估、看涨
                    "peer_upside": float(np.log(peer["peer_fv"] / cur_close)) if peer.get("peer_fv") else np.nan,
                    "stat_upside": float(np.log(stat_fv / cur_close)) if stat_fv else np.nan,
                    "comp_upside": float(np.log(comp_fv / cur_close)),
                    "z_score": float(np.log(cur_close / comp_fv) / sigma_1d),
                    "ret_1d": float(nxt_1 / cur_close - 1.0),
                    "ret_3d": float(nxt_3 / cur_close - 1.0),
                    "ret_5d": float(nxt_5 / cur_close - 1.0),
                    "in_1s": bool(low_1s <= nxt_1 <= high_1s),
                    "in_90": bool(low_90 <= nxt_1 <= high_90),
                }
            )

    if not records:
        return {"error": "回测未产生有效样本"}

    rdf = pd.DataFrame(records)

    def _layer_metrics(col: str) -> dict[str, Any]:
        sub = rdf.dropna(subset=[col])
        out: dict[str, Any] = {"samples": int(len(sub))}
        for h in (1, 3, 5):
            rcol = f"ret_{h}d"
            x = sub[col].to_numpy(dtype=float)
            y = sub[rcol].to_numpy(dtype=float)
            out[f"ic_{h}d"] = round(_pearson_corr(x, y), 4)
            out[f"rank_ic_{h}d"] = round(_spearman_corr(x, y), 4)
        return out

    # 五分位多空分组（最被低估前 20% vs 最被高估后 20%，标准机构因子检验口径）
    q20 = float(rdf["z_score"].quantile(0.20))
    q80 = float(rdf["z_score"].quantile(0.80))
    under = rdf[rdf["z_score"] <= q20]
    over = rdf[rdf["z_score"] >= q80]
    buckets = {
        "q20_z_threshold": round(q20, 2),
        "q80_z_threshold": round(q80, 2),
        "undervalued_count": int(len(under)),
        "undervalued_avg_ret_3d_pct": round(float(under["ret_3d"].mean() * 100.0), 3) if len(under) else None,
        "undervalued_hit_3d_pct": round(float((under["ret_3d"] > 0).mean() * 100.0), 1) if len(under) else None,
        "overvalued_count": int(len(over)),
        "overvalued_avg_ret_3d_pct": round(float(over["ret_3d"].mean() * 100.0), 3) if len(over) else None,
        "overvalued_hit_3d_pct": round(float((over["ret_3d"] < 0).mean() * 100.0), 1) if len(over) else None,
        "long_short_spread_3d_pct": (
            round(float((under["ret_3d"].mean() - over["ret_3d"].mean()) * 100.0), 3)
            if len(under) and len(over)
            else None
        ),
        "long_short_spread_5d_pct": (
            round(float((under["ret_5d"].mean() - over["ret_5d"].mean()) * 100.0), 3)
            if len(under) and len(over)
            else None
        ),
    }

    # 分板块表现
    by_group: dict[str, Any] = {}
    for g, gdf in rdf.groupby("group"):
        gq20 = float(gdf["z_score"].quantile(0.20))
        gq80 = float(gdf["z_score"].quantile(0.80))
        gu = gdf[gdf["z_score"] <= gq20]
        go = gdf[gdf["z_score"] >= gq80]
        by_group[str(g)] = {
            "samples": int(len(gdf)),
            "rank_ic_3d": round(_spearman_corr(gdf["comp_upside"].to_numpy(), gdf["ret_3d"].to_numpy()), 4),
            "rank_ic_5d": round(_spearman_corr(gdf["comp_upside"].to_numpy(), gdf["ret_5d"].to_numpy()), 4),
            "ls_spread_3d_pct": (
                round(float((gu["ret_3d"].mean() - go["ret_3d"].mean()) * 100.0), 3)
                if len(gu) >= 5 and len(go) >= 5
                else None
            ),
            "in_1s_pct": round(float(gdf["in_1s"].mean() * 100.0), 1),
        }

    return {
        "period": f"{rdf['date'].min().date()} → {rdf['date'].max().date()}",
        "trading_days": int(rdf["date"].nunique()),
        "symbols_evaluated": int(rdf["symbol"].nunique()),
        "total_samples": int(len(rdf)),
        "layers": {
            "peer": _layer_metrics("peer_upside"),
            "stat": _layer_metrics("stat_upside"),
            "composite": _layer_metrics("comp_upside"),
        },
        "buckets_z1": buckets,
        "range_calibration": {
            "in_1s_pct": round(float(rdf["in_1s"].mean() * 100.0), 1),
            "target_1s_pct": 68.3,
            "in_90_pct": round(float(rdf["in_90"].mean() * 100.0), 1),
            "target_90_pct": 90.0,
        },
        "by_group": by_group,
    }
