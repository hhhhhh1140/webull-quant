"""趋势研判回测：上涨趋势只做 SOXL，下跌趋势只做 SOXS，过渡期空仓（现金吃短债利息）。

判断全部基于费城半导体指数 ^SOX（1 倍）而不是 3 倍 ETF 本身，避免 3 倍波动的噪音。
所有信号只用第 t 天收盘及之前的数据，第 t+1 天开盘成交。

数据：
  1995-2010-03 SOXL/SOXS 尚未上市 → 用 ^SOX 日收益 ×(±3) 减去融资和管理费合成（参数用真实数据校准）
  2010-03 起 → 用 SOXL / SOXS 真实复权价格

研判由四类信号组成（每类只用当时可得的数据）：
  1. K 线趋势（闸门）：收盘 vs 200 日均线 + 200 日均线斜率 → 决定只能做多 / 只能做空 / 空仓
  2. K 线动量：50 日均线、3 个月涨跌
  3. 利率周期：13 周美债收益率（紧跟联邦基金利率）6 个月变化 → 加息 / 降息 / 紧急降息；
     10 年期美债 3 个月急升 → 利率冲击
  4. 风险情绪（消息面的市场化代理）：VIX 水平和相对 50 日均值的跳升
     —— 历史新闻没法无前视地回测，VIX 是新闻冲击最直接的定价结果
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import regime_data

COST = 0.0010          # 每单边点差 + 滑点（占成交金额）
FEE_L = 0.0095         # SOXL/SOXS 管理费率（年）
START = "1995-06-01"
VOL_TARGET = 0.25     # 1 倍指数的目标年化波动率


# ---------------------------------------------------------------- 数据和合成 3 倍收益
def build_frame(d: dict | None = None) -> pd.DataFrame:
    d = d or regime_data.load()
    sox = d["SOX"]
    f = pd.DataFrame({"o": sox.open, "h": sox.high, "l": sox.low, "c": sox.close})
    f = f[(f.c > 0)]
    f["o"] = f.o.where(f.o > 0, f.c.shift())
    for n in ("VIX", "IRX", "FVX", "TNX"):
        f[n.lower()] = d[n].close.reindex(f.index).ffill()
    f["cash"] = f.irx.fillna(0).clip(lower=0) / 100 / 252             # 现金日收益
    prev = f.c.shift()
    f["r_on"] = f.o / prev - 1           # 隔夜（昨收→今开）
    f["r_id"] = f.c / f.o - 1            # 日内（今开→今收）
    f["r"] = f.c / prev - 1

    # 合成 3 倍：做多要借 2 倍资金（短债利率 + 利差），做空持有现金但要付掉期利差
    drag_L = (FEE_L + 2 * (f.irx.fillna(0) / 100 + 0.004)) / 252
    drag_S = (FEE_L + 0.004 * 3) / 252 - f.cash * 0  # 反向基金现金利息基本被掉期成本吃掉
    L_on, L_id = 3 * f.r_on, 3 * f.r_id - drag_L
    S_on, S_id = -3 * f.r_on, -3 * f.r_id - drag_S

    # 真实 SOXL / SOXS（复权）
    for k, sgn, on, idd in (("SOXL", 1, L_on, L_id), ("SOXS", -1, S_on, S_id)):
        e = d[k].reindex(f.index)
        adj = e.adjclose / e.close
        ao, ac = e.open * adj, e.adjclose
        real_on, real_id = ao / ac.shift() - 1, ac / ao - 1
        ok = real_on.notna() & real_id.notna() & (e.open > 0)
        f[f"{k}_on"] = real_on.where(ok, on)
        f[f"{k}_id"] = real_id.where(ok, idd)
        f[f"{k}_real"] = ok
    return f.loc[START:].dropna(subset=["r"])


def tracking_report(f: pd.DataFrame) -> pd.DataFrame:
    """合成 3 倍 vs 真实 ETF（2010-03 起）：校验合成方法是否靠谱。"""
    rows = []
    for k, sgn in (("SOXL", 3), ("SOXS", -3)):
        m = f[f[f"{k}_real"]]
        real = (1 + m[f"{k}_on"]) * (1 + m[f"{k}_id"]) - 1
        syn = sgn * m.r - (FEE_L + (2 * (m.irx / 100 + .004) if sgn > 0 else .012)) / 252
        yrs = len(m) / 252
        rows.append({"ETF": k, "相关系数": real.corr(syn),
                     "真实年化": (1 + real).prod() ** (1 / yrs) - 1,
                     "合成年化": (1 + syn).prod() ** (1 / yrs) - 1})
    return pd.DataFrame(rows).set_index("ETF")


# ---------------------------------------------------------------- 信号
def features(f: pd.DataFrame, sma_long: int = 200, sma_mid: int = 50) -> pd.DataFrame:
    c = f.c
    x = pd.DataFrame(index=f.index)
    x["sma_l"] = c.rolling(sma_long).mean()
    x["sma_m"] = c.rolling(sma_mid).mean()
    x["slope"] = x.sma_l / x.sma_l.shift(20) - 1
    x["mom63"] = c / c.shift(63) - 1
    x["vol20"] = f.r.rolling(20).std() * np.sqrt(252)
    # 闸门：上涨趋势 / 下跌趋势 / 过渡
    x["regime"] = np.select([(c > x.sma_l) & (x.slope > 0), (c < x.sma_l) & (x.slope < 0)], [1, -1], 0)
    x["mom"] = np.select([(c > x.sma_m) & (x.mom63 > 0), (c < x.sma_m) & (x.mom63 < 0)], [1, -1], 0)
    # 利率周期
    irx_6m = f.irx - f.irx.shift(126)
    irx_3m = f.irx - f.irx.shift(63)
    tnx_3m = f.tnx - f.tnx.shift(63)
    x["hiking"] = irx_6m > 0.5
    x["cutting"] = irx_6m < -0.5
    x["emergency"] = irx_3m < -1.0                  # 3 个月短端利率掉 100bp：危机式降息
    x["rate_shock"] = tnx_3m > 0.6                  # 长端 3 个月急升 60bp
    x["rates"] = np.select([x.hiking | x.rate_shock | x.emergency, x.cutting], [-1, 1], 0)
    # 风险情绪（VIX）
    vsma = f.vix.rolling(50).mean()
    x["stress"] = (f.vix > 30) | (f.vix > 1.3 * vsma)
    x["calm"] = (f.vix < 20) & (f.vix < vsma)
    x["risk"] = np.select([x.stress, x.calm], [-1, 1], 0)
    x["score"] = 2 * x.regime + x.mom + x.rates + x.risk
    return x


# ---------------------------------------------------------------- 策略：返回目标仓位（+ 为 SOXL 比例，- 为 SOXS 比例）
def strategies(f: pd.DataFrame, x: pd.DataFrame) -> dict[str, pd.Series]:
    c, z = f.c, pd.Series(0.0, index=f.index)
    up, dn = (x.regime == 1).astype(float), (x.regime == -1).astype(float)
    # 波动率控制：1 倍指数 20 日波动率高于 25% 时按比例降仓（3 倍后约 75%）
    scale = (VOL_TARGET / x.vol20).clip(upper=1.0).fillna(0)
    # 利率权重：温和降息期满仓，加息 / 危机式降息 / 长端急升期 60%，其余 80%
    macro = pd.Series(np.select([x.cutting & ~x.emergency, x.hiking | x.emergency | x.rate_shock],
                                [1.0, 0.6], 0.8), index=f.index)
    S = {}
    S["持有SOXL"] = z + 1.0
    S["均线200 多空翻转"] = pd.Series(np.where(c > x.sma_l, 1.0, -1.0), index=f.index)
    S["趋势闸门 多空"] = up - dn
    S["趋势闸门 多空+控波动"] = (up - dn) * scale
    # 综合研判：趋势 ±2 + 动量 ±1 + 利率 ±1 + VIX ±1；趋势闸门决定方向，分数决定仓位
    sc = x.score
    longw = np.where(x.regime == 1, np.select([sc >= 4, sc >= 3], [1.0, 0.5], 0.0), 0.0)
    shortw = np.where(x.regime == -1, np.select([sc <= -4, sc <= -3], [1.0, 0.5], 0.0), 0.0)
    S["综合打分 多空"] = pd.Series(longw - shortw, index=f.index)
    S["趋势闸门 只做多"] = up
    S["只做多+控波动"] = up * scale
    S["只做多+控波动+利率权重"] = up * scale * macro
    for k in S:
        if k != "持有SOXL":
            S[k] = S[k].where(x.sma_l.notna(), 0.0)
    return S


# ---------------------------------------------------------------- 回测引擎：第 t 天收盘出信号，第 t+1 天开盘调仓
def run(f: pd.DataFrame, target: pd.Series, cost: float = COST) -> pd.DataFrame:
    tgt = target.shift(1).fillna(0)                 # 今天开盘执行昨天收盘的信号
    held = tgt.shift(1).fillna(0)                   # 隔夜段持有的是前一天的仓位
    def leg(w, on):
        L, S = (f.SOXL_on, f.SOXS_on) if on else (f.SOXL_id, f.SOXS_id)
        return w.clip(lower=0) * L + (-w.clip(upper=0)) * S
    cash_w_on, cash_w_id = 1 - held.abs(), 1 - tgt.abs()
    r_on = leg(held, True)
    # 开盘调仓：多/空腿各自按成交金额算成本（换边 = 卖一只 + 买另一只）
    turn = (tgt.clip(lower=0) - held.clip(lower=0)).abs() + (tgt.clip(upper=0) - held.clip(upper=0)).abs()
    r_id = leg(tgt, False) + cash_w_id * f.cash
    ret = (1 + r_on) * (1 + r_id) - 1 - turn * cost
    return pd.DataFrame({"ret": ret, "pos": tgt, "turn": turn})


def stats(ret: pd.Series, pos: pd.Series | None = None, turn: pd.Series | None = None) -> dict:
    ret = ret.dropna()
    if len(ret) < 20:
        return {}
    eq = (1 + ret).cumprod()
    yrs = len(ret) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    vol = ret.std() * np.sqrt(252)
    out = {"年化": cagr, "最大回撤": dd, "夏普": (ret.mean() * 252) / vol if vol else np.nan,
           "收益回撤比": cagr / -dd if dd else np.nan, "总倍数": eq.iloc[-1]}
    if pos is not None:
        p = pos.loc[ret.index]
        out["做多天数%"] = (p > 0).mean()
        out["做空天数%"] = (p < 0).mean()
        out["每年换手"] = turn.loc[ret.index].sum() / yrs if turn is not None else np.nan
    return out


PERIODS = {
    "全期 1996-2026": ("1996-01-01", None),
    "合成期 1996-2009": ("1996-01-01", "2009-12-31"),
    "真实ETF 2010-2017": ("2010-03-12", "2017-12-31"),
    "真实ETF 2018-2026": ("2018-01-01", None),
    "2022 加息熊市": ("2022-01-01", "2022-12-31"),
    "2026 年至今": ("2026-01-01", None),
}


def period_table(f, S, periods=PERIODS) -> dict[str, pd.DataFrame]:
    out = {}
    for pname, (a, b) in periods.items():
        rows = {}
        for k, t in S.items():
            bt = run(f, t)
            sl = slice(a, b)
            rows[k] = stats(bt.ret.loc[sl], bt.pos, bt.turn)
        out[pname] = pd.DataFrame(rows).T
    return out


def regime_breakdown(f, x, S, a="1996-01-01") -> pd.DataFrame:
    """按利率周期拆开看：加息 / 降息 / 紧急降息 / 平稳期 各自的年化。"""
    lab = pd.Series("平稳", index=f.index)
    lab[x.hiking] = "加息期"
    lab[x.cutting] = "降息期"
    lab[x.emergency] = "危机式降息"
    rows = {}
    for k in ("持有SOXL", "趋势闸门 多空", "综合打分 多空", "只做多+控波动", "只做多+控波动+利率权重"):
        bt = run(f, S[k]).loc[a:]
        g = bt.ret.groupby(lab.loc[a:])
        rows[k] = (1 + g.mean()) ** 252 - 1
    sox = f.r.loc[a:].groupby(lab.loc[a:])
    rows["SOX指数(1倍)"] = (1 + sox.mean()) ** 252 - 1
    t = pd.DataFrame(rows).T
    t.loc["天数"] = lab.loc[a:].value_counts()
    return t


def sensitivity(f) -> pd.DataFrame:
    """参数敏感性：换长均线和目标波动率，看「只做多+控波动」是不是只在某一组参数上有效。"""
    rows = []
    for L in (100, 150, 200, 250):
        x = features(f, L)
        for tv in (0.20, 0.25, 0.30):
            t = ((x.regime == 1) * (tv / x.vol20).clip(upper=1.0)).fillna(0).where(x.sma_l.notna(), 0.0)
            bt = run(f, t)
            for pname, (a, b) in list(PERIODS.items())[1:4]:
                s = stats(bt.ret.loc[a:b])
                rows.append({"长均线": L, "目标波动": f"{tv:.0%}", "区间": pname, "年化": s["年化"], "回撤": s["最大回撤"]})
    return pd.DataFrame(rows).pivot_table(index=["长均线", "目标波动"], columns="区间", values=["年化", "回撤"], sort=False)


def yearly(f, S, keys=("持有SOXL", "趋势闸门 多空", "只做多+控波动", "只做多+控波动+利率权重")) -> pd.DataFrame:
    t = pd.DataFrame({k: run(f, S[k]).ret.groupby(f.index.year).apply(lambda r: (1 + r).prod() - 1) for k in keys})
    sh = run(f, S["趋势闸门 多空"].clip(upper=0)).ret
    t["其中做空腿"] = sh.groupby(f.index.year).apply(lambda r: (1 + r).prod() - 1)
    t["SOX指数"] = f.r.groupby(f.index.year).apply(lambda r: (1 + r).prod() - 1)
    return t


def current_view(f, x) -> dict:
    i = x.index[-1]
    r = x.loc[i]
    return {"日期": i.date().isoformat(), "SOX": round(f.c.iloc[-1], 1),
            "200日线": round(r.sma_l, 1), "50日线": round(r.sma_m, 1),
            "200日线20日斜率": f"{r.slope:+.2%}", "3个月涨跌": f"{r.mom63:+.1%}",
            "趋势闸门": {1: "上涨(只做多)", -1: "下跌(只做空)", 0: "过渡(空仓)"}[int(r.regime)],
            "动量": int(r.mom), "利率": int(r.rates), "VIX情绪": int(r.risk), "综合分": int(r.score),
            "13周美债": round(f.irx.iloc[-1], 2), "13周美债6个月变化": round(f.irx.iloc[-1] - f.irx.iloc[-127], 2),
            "10年美债3个月变化": round(f.tnx.iloc[-1] - f.tnx.iloc[-64], 2),
            "VIX": round(f.vix.iloc[-1], 1), "20日波动率": f"{r.vol20:.0%}"}


def main(fetch: bool = False) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 20)
    if fetch:
        regime_data.fetch_all()
    f = build_frame()
    x = features(f)
    S = strategies(f, x)
    print("== 合成 3 倍 vs 真实 ETF（2010-03 起）==")
    print(tracking_report(f).to_string(float_format=lambda v: f"{v:.3f}"))
    fmt = {"年化": "{:+.1%}", "最大回撤": "{:.1%}", "夏普": "{:.2f}", "收益回撤比": "{:.2f}",
           "总倍数": "{:,.1f}", "做多天数%": "{:.0%}", "做空天数%": "{:.0%}", "每年换手": "{:.1f}"}
    for pname, t in period_table(f, S).items():
        print(f"\n== {pname} ==")
        print(t.apply(lambda col: col.map(lambda v, c=col.name: fmt.get(c, "{}").format(v))).to_string())
    print("\n== 逐年收益 ==")
    print(yearly(f, S).to_string(float_format=lambda v: f"{v:+.0%}"))
    print("\n== 按利率周期拆分（年化，1996 起）==")
    print(regime_breakdown(f, x, S).to_string(float_format=lambda v: f"{v:+.1%}" if abs(v) < 50 else f"{v:.0f}"))
    print("\n== 参数敏感性：只做多+控波动 ==")
    print(sensitivity(f).to_string(float_format=lambda v: f"{v:+.1%}"))
    print("\n== 当前研判 ==")
    for k, v in current_view(f, x).items():
        print(f"  {k}: {v}")
