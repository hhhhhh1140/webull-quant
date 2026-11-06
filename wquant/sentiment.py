"""情绪 / 市场环境指标。scan 时计算，写进候选包，供 Claude 写"市场研判"时参考。

- 市场广度：股票池里站在 SMA20 / SMA50 上方的比例，以及 5 日变化（只用价格，可回测）
- 指数位置：SPY / QQQ 相对 SMA20/50/200 的距离、20 日涨跌
- 板块涨跌家数（Webull 板块接口）
- 大单资金流向（Webull 资金流接口）
- 恐惧贪婪指数（Webull，沙盒不提供，取不到就留空）
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def breadth(frames: dict[str, pd.DataFrame], symbols: list[str], at: pd.Timestamp | None = None) -> dict:
    """symbols 里站在 SMA20 / SMA50 上方的比例。at=None 取最后一行。"""
    above20 = above50 = n = 0
    for s in symbols:
        df = frames.get(s)
        if df is None or df.empty:
            continue
        sub = df if at is None else df.loc[:at]
        if sub.empty:
            continue
        row = sub.iloc[-1]
        if pd.isna(row.get("sma_fast")) or pd.isna(row.get("st_fast")):
            continue
        n += 1
        above20 += row["close"] > row["st_fast"]
        above50 += row["close"] > row["sma_fast"]
    if not n:
        return {"n": 0, "above_sma20": None, "above_sma50": None}
    return {"n": n, "above_sma20": round(above20 / n, 3), "above_sma50": round(above50 / n, 3)}


def breadth_symbols(settings) -> list[str]:
    """参与广度计算的股票：池子里的普通股票（去掉指数和杠杆/反向 ETF）。"""
    return [s for s, g in settings.groups.items()
            if g != "index" and settings.instrument(s)["leverage"] <= 1 and not settings.instrument(s)["inverse"]]


def index_snapshot(frames: dict[str, pd.DataFrame], symbols: list[str]) -> list[dict]:
    out = []
    for s in symbols:
        df = frames.get(s)
        if df is None or len(df) < 21:
            continue
        r = df.iloc[-1]
        pct = lambda col: None if pd.isna(r[col]) else round(float(r["close"] / r[col] - 1), 4)
        out.append({"symbol": s, "close": round(float(r["close"]), 2), "vs_sma20": pct("st_fast"),
                    "vs_sma50": pct("sma_fast"), "vs_sma200": pct("sma_slow"),
                    "ret_5d": round(float(df["close"].iloc[-1] / df["close"].iloc[-6] - 1), 4),
                    "ret_20d": round(float(df["close"].iloc[-1] / df["close"].iloc[-21] - 1), 4),
                    "rsi14": round(float(r["rsi"]), 1)})
    return out


def _webull_extras(client, cfg: dict) -> dict:
    """接口取不到的部分留空，不影响流程。"""
    from .webull_client import _json, _unwrap_list
    out: dict = {"sectors": None, "capital_flow": None, "fear_greed": None}
    try:
        rows = _unwrap_list(_json(client.data.screener.get_market_sectors("US_STOCK", page_size=30)))
        keep = [k.lower() for k in cfg.get("sectors", ["Semiconductor", "Software", "Technology"])]
        out["sectors"] = [
            {"name": r.get("name"), "change": float(r.get("change_ratio") or 0),
             "advanced": int(r.get("advanced") or 0), "declined": int(r.get("declined") or 0)}
            for r in rows if any(k in (r.get("name") or "").lower() for k in keep)
        ]
    except Exception as e:
        log.info("板块数据不可用: %s", e)
    flows = {}
    for sym in cfg.get("flow_symbols", ["QQQ", "SOXL"]):
        try:
            rows = _unwrap_list(_json(client.data.fundamentals.get_capital_flow(sym, count=5)))
            net = [round((float(r.get("large_in", 0)) - float(r.get("large_out", 0))) / 1e6, 1) for r in rows]
            flows[sym] = {"dates": [r.get("date") for r in rows], "large_net_musd": net}
        except Exception as e:
            log.info("%s 资金流不可用: %s", sym, e)
    out["capital_flow"] = flows or None
    try:
        out["fear_greed"] = _json(client.data.screener.get_fear_and_greed_index())
    except Exception as e:
        log.info("恐惧贪婪指数不可用: %s", e)
    return out


def compute(settings, frames: dict[str, pd.DataFrame], client=None) -> dict:
    cfg = settings.section("sentiment")
    syms = breadth_symbols(settings)
    now = breadth(frames, syms)
    # 5 个交易日前的广度，看变化速度
    ref = next((f for s, f in frames.items() if s in syms and len(f) > 6), None)
    prev = breadth(frames, syms, at=ref.index[-6]) if ref is not None else {}
    change = None
    if now.get("above_sma50") is not None and prev.get("above_sma50") is not None:
        change = round(now["above_sma50"] - prev["above_sma50"], 3)
    idx = (settings.section("strategy").get("market_filter") or {}).get("symbols", ["SPY", "QQQ"])
    out = {
        "breadth": {**now, "above_sma50_5d_ago": prev.get("above_sma50"), "above_sma50_change_5d": change},
        "indices": index_snapshot(frames, idx),
        "by_group": {},
    }
    for g in sorted(set(settings.groups.values())):
        members = [s for s in syms if settings.groups[s] == g]
        if members:
            out["by_group"][g] = breadth(frames, members)
    if client is not None:
        out.update(_webull_extras(client, cfg))
    return out
