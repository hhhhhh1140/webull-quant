"""三层公允价值系统的数据层：日线行情、盘前快照、基本面估值倍数与事件日历。

- 日线行情支持四种来源：yahoo（默认，免密钥且自动前复权）、webull、cache、synthetic
- 基本面与盘前报价来自 Yahoo Finance quoteSummary（免费公开接口），按日缓存在 data/fair_value/
- 宏观与财报窗口联动项目已有的 events.yaml 与 state/earnings.json
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import ROOT, Settings
from .data import load_bars as load_existing_bars
from .data import load_csv, synthetic_bars
from .events import all_events

log = logging.getLogger(__name__)

FV_DATA_DIR = ROOT / "data" / "fair_value"
FV_BARS_DIR = FV_DATA_DIR / "bars"
FV_FUND_DIR = FV_DATA_DIR / "fundamentals"

YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{}"
YAHOO_SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{}"
YAHOO_CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"
YAHOO_MODULES = "financialData,defaultKeyStatistics,summaryDetail,price,calendarEvents"

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"


def _raw(obj: Any) -> float | None:
    """从 Yahoo Finance 字段（可能是 {'raw': 1.23, 'fmt': ...} 或标量）提取 float。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        obj = obj.get("raw")
    if obj is None or isinstance(obj, bool):
        return None
    try:
        v = float(obj)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _make_yahoo_session() -> tuple[requests.Session, str]:
    """创建带 cookie + crumb 的 Yahoo Finance 会话。"""
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    try:
        sess.get("https://fc.yahoo.com", timeout=12)
    except Exception:
        pass
    crumb = ""
    for attempt in range(3):
        try:
            r = sess.get(YAHOO_CRUMB_URL, timeout=12)
            if r.status_code == 200 and r.text and "Too Many Requests" not in r.text:
                crumb = r.text.strip()
                break
        except Exception as e:
            log.debug("获取 Yahoo crumb 第 %d 次失败: %s", attempt + 1, e)
        time.sleep(1.0 * (attempt + 1))
    return sess, crumb


def fetch_yahoo_bars(
    symbol: str,
    years: int = 5,
    session: requests.Session | None = None,
    crumb: str = "",
) -> pd.DataFrame:
    """拉取单个标的的 Yahoo 前复权日线（OHLCV）。"""
    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", USER_AGENT)
    end_ts = int(time.time()) + 86400
    start_ts = int((pd.Timestamp.today() - pd.DateOffset(years=years)).timestamp())
    params: dict[str, Any] = {"period1": start_ts, "period2": end_ts, "interval": "1d", "events": "div,split"}
    if crumb:
        params["crumb"] = crumb

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            r = sess.get(YAHOO_CHART_URL.format(symbol), params=params, timeout=20)
            if r.status_code == 200:
                payload = r.json()
                result = (payload.get("chart") or {}).get("result") or []
                if not result:
                    raise ValueError(f"Yahoo chart 无返回数据: {symbol}")
                node = result[0]
                ts = node.get("timestamp") or []
                q = (node.get("indicators") or {}).get("quote", [{}])[0]
                adj_list = ((node.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
                idx = (
                    pd.to_datetime(ts, unit="s", utc=True)
                    .tz_convert("America/New_York")
                    .normalize()
                    .tz_localize(None)
                )
                df = pd.DataFrame(
                    {
                        "open": q.get("open"),
                        "high": q.get("high"),
                        "low": q.get("low"),
                        "close": q.get("close"),
                        "volume": q.get("volume"),
                    },
                    index=idx,
                ).astype(float)
                if adj_list is not None and len(adj_list) == len(df):
                    adj = pd.Series(adj_list, index=idx, dtype=float)
                    valid = (df["close"] > 0) & adj.notna() & (adj > 0)
                    ratio = pd.Series(1.0, index=idx)
                    ratio.loc[valid] = adj.loc[valid] / df.loc[valid, "close"]
                    for col in ("open", "high", "low", "close"):
                        df[col] = df[col] * ratio
                df = df.dropna(subset=["close"])
                df["volume"] = df["volume"].fillna(0.0)
                df = df[~df.index.duplicated(keep="last")].sort_index()
                df.index.name = "date"
                return df
            last_err = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"拉取 {symbol} Yahoo 日线失败: {last_err}")


def _save_fv_bars(symbol: str, df: pd.DataFrame) -> None:
    FV_BARS_DIR.mkdir(parents=True, exist_ok=True)
    p = FV_BARS_DIR / f"{symbol.upper()}.csv"
    if p.exists():
        try:
            old = load_csv(p)
            df = pd.concat([old[~old.index.isin(df.index)], df]).sort_index()
        except Exception:
            pass
    df.to_csv(p, index_label="date")


def load_fv_bars(
    settings: Settings,
    symbols: list[str],
    source: str = "yahoo",
    years: int = 5,
    refresh: bool = True,
) -> dict[str, pd.DataFrame]:
    """统一加载日线。source: yahoo | webull | cache | synthetic。"""
    syms = list(dict.fromkeys(s.upper() for s in symbols))
    if source == "synthetic":
        return {s: synthetic_bars(s, n=min(years * 252, 1200)) for s in syms}
    if source == "webull":
        return load_existing_bars(settings, syms, source="webull", count=min(years * 252, 1200), refresh=refresh)
    if source == "cache":
        out: dict[str, pd.DataFrame] = {}
        for s in syms:
            p_fv = FV_BARS_DIR / f"{s}.csv"
            p_main = ROOT / "data" / "cache" / f"{s}.csv"
            if p_fv.exists():
                out[s] = load_csv(p_fv)
            elif p_main.exists():
                out[s] = load_csv(p_main)
        return out

    # source == "yahoo"
    sess, crumb = _make_yahoo_session()
    out = {}
    consec_fail = 0
    for s in syms:
        cached_path = FV_BARS_DIR / f"{s}.csv"
        fallback_path = ROOT / "data" / "cache" / f"{s}.csv"
        if not refresh or consec_fail >= 3:
            if cached_path.exists():
                out[s] = load_csv(cached_path)
            elif fallback_path.exists():
                out[s] = load_csv(fallback_path)
            continue
        try:
            df = fetch_yahoo_bars(s, years=years, session=sess, crumb=crumb)
            _save_fv_bars(s, df)
            out[s] = df
            consec_fail = 0
            time.sleep(0.15)
        except Exception as e:
            consec_fail += 1
            log.warning("%s 拉取 Yahoo 日线失败，尝试回退本地缓存: %s", s, e)
            if cached_path.exists():
                out[s] = load_csv(cached_path)
            elif fallback_path.exists():
                out[s] = load_csv(fallback_path)
    return out


def parse_yahoo_fundamental(symbol: str, payload: dict[str, Any]) -> dict[str, Any]:
    """把 Yahoo quoteSummary JSON 规整为统一的基本面 + 实时/盘前报价字典。"""
    price = payload.get("price") or {}
    sd = payload.get("summaryDetail") or {}
    ks = payload.get("defaultKeyStatistics") or {}
    fd = payload.get("financialData") or {}
    cal = (payload.get("calendarEvents") or {}).get("earnings") or {}

    earnings_dates = []
    for item in cal.get("earningsDate") or []:
        if isinstance(item, dict) and item.get("fmt"):
            earnings_dates.append(str(item["fmt"]))
        elif isinstance(item, dict) and item.get("raw"):
            earnings_dates.append(datetime.utcfromtimestamp(item["raw"]).strftime("%Y-%m-%d"))

    quote_type = str(price.get("quoteType") or "").upper()
    market_state = str(price.get("marketState") or "").upper()
    reg_price = _raw(price.get("regularMarketPrice")) or _raw(fd.get("currentPrice"))
    prev_close = _raw(price.get("regularMarketPreviousClose"))
    pre_price = _raw(price.get("preMarketPrice"))
    post_price = _raw(price.get("postMarketPrice"))

    # 选择最能反映开盘前/当下市场共识的参考价
    if market_state == "PRE" and pre_price:
        ref_price, ref_source = pre_price, "pre_market"
    elif market_state in ("POST", "POSTPOST") and post_price:
        ref_price, ref_source = post_price, "post_market"
    elif pre_price and market_state == "PREPRE":
        ref_price, ref_source = pre_price, "pre_market"
    elif reg_price:
        ref_price, ref_source = reg_price, "regular"
    else:
        ref_price, ref_source = prev_close, "prev_close"

    return {
        "symbol": symbol.upper(),
        "fetched_date": str(date.today()),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "quote_type": quote_type or "EQUITY",
        "market_state": market_state,
        "regular_price": reg_price,
        "prev_close": prev_close,
        "pre_market_price": pre_price,
        "post_market_price": post_price,
        "ref_price": ref_price,
        "ref_source": ref_source,
        "market_cap": _raw(price.get("marketCap")),
        "beta": _raw(sd.get("beta")) or _raw(ks.get("beta")),
        "trailing_pe": _raw(sd.get("trailingPE")) or _raw(ks.get("trailingPE")),
        "forward_pe": _raw(ks.get("forwardPE")) or _raw(sd.get("forwardPE")),
        "trailing_eps": _raw(ks.get("trailingEps")),
        "forward_eps": _raw(ks.get("forwardEps")),
        "peg_ratio": _raw(ks.get("pegRatio")),
        "price_to_sales": _raw(sd.get("priceToSalesTrailing12Months")),
        "ev_to_ebitda": _raw(ks.get("enterpriseToEbitda")),
        "price_to_book": _raw(ks.get("priceToBook")),
        "revenue_growth": _raw(fd.get("revenueGrowth")),
        "earnings_growth": _raw(fd.get("earningsGrowth")),
        "profit_margins": _raw(fd.get("profitMargins")),
        "roe": _raw(fd.get("returnOnEquity")),
        "target_mean": _raw(fd.get("targetMeanPrice")),
        "target_median": _raw(fd.get("targetMedianPrice")),
        "target_low": _raw(fd.get("targetLowPrice")),
        "target_high": _raw(fd.get("targetHighPrice")),
        "analyst_count": int(_raw(fd.get("numberOfAnalystOpinions")) or 0),
        "recommendation_mean": _raw(fd.get("recommendationMean")),
        "earnings_dates": earnings_dates,
    }


def _fund_cache_path(symbol: str) -> Path:
    return FV_FUND_DIR / f"{symbol.upper()}.json"


def load_fundamentals(
    symbols: list[str],
    refresh: bool = True,
    max_age_days: int = 0,
) -> dict[str, dict[str, Any]]:
    """拉取并缓存基本面与盘前报价。refresh=False 时仅读取本地缓存。"""
    syms = list(dict.fromkeys(s.upper() for s in symbols))
    today = date.today()
    out: dict[str, dict[str, Any]] = {}
    to_fetch: list[str] = []

    for s in syms:
        p = _fund_cache_path(s)
        cached = None
        if p.exists():
            try:
                cached = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                cached = None
        if cached and not refresh:
            out[s] = cached
            continue
        if cached and max_age_days > 0:
            try:
                age = (today - date.fromisoformat(cached.get("fetched_date", "1970-01-01"))).days
                if age <= max_age_days:
                    out[s] = cached
                    continue
            except Exception:
                pass
        if cached:
            out[s] = cached  # 作为网络失败时的保底
        to_fetch.append(s)

    if not to_fetch or not refresh:
        return out

    FV_FUND_DIR.mkdir(parents=True, exist_ok=True)
    sess, crumb = _make_yahoo_session()
    for s in to_fetch:
        params: dict[str, str] = {"modules": YAHOO_MODULES}
        if crumb:
            params["crumb"] = crumb
        for attempt in range(2):
            try:
                r = sess.get(YAHOO_SUMMARY_URL.format(s), params=params, timeout=15)
                if r.status_code == 200:
                    res = (r.json().get("quoteSummary") or {}).get("result") or []
                    if res:
                        parsed = parse_yahoo_fundamental(s, res[0])
                        _fund_cache_path(s).write_text(
                            json.dumps(parsed, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        out[s] = parsed
                        break
                elif r.status_code in (401, 403) and attempt == 0:
                    sess, crumb = _make_yahoo_session()
                    if crumb:
                        params["crumb"] = crumb
                    continue
            except Exception as e:
                log.debug("拉取 %s 基本面失败: %s", s, e)
            time.sleep(0.5)
    return out


def load_event_context(
    fundamentals: dict[str, dict[str, Any]] | None = None,
    as_of: date | None = None,
    earnings_window_days: int = 3,
    macro_window_hours: int = 36,
) -> dict[str, Any]:
    """整合 events.yaml（宏观事件）、state/earnings.json 与 Yahoo 财报日历。

    返回:
      {
        "macro_alerts": [str, ...],
        "earnings_flags": {symbol: {"status": "EARNINGS_SOON" | "POST_EARNINGS_REPRICING", "date": str, "bdays": int}}
      }
    """
    ref_day = as_of or date.today()
    now_dt = datetime.combine(ref_day, datetime.min.time())
    events = all_events()

    macro_alerts: list[str] = []
    earnings_map: dict[str, list[date]] = {}

    for ev in events:
        try:
            ev_day = date.fromisoformat(str(ev["date"]))
        except Exception:
            continue
        if ev.get("type") == "macro":
            delta_days = (ev_day - ref_day).days
            if 0 <= delta_days <= max(1, int(np.ceil(macro_window_hours / 24))):
                if ev.get("importance", "high") == "high":
                    macro_alerts.append(f"{ev.get('name', '宏观事件')} ({ev['date']})")
        elif ev.get("type") == "earnings" and ev.get("symbol"):
            earnings_map.setdefault(str(ev["symbol"]).upper(), []).append(ev_day)

    for sym, fd in (fundamentals or {}).items():
        for ds in fd.get("earnings_dates") or []:
            try:
                earnings_map.setdefault(sym.upper(), []).append(date.fromisoformat(ds))
            except Exception:
                continue

    earnings_flags: dict[str, dict[str, Any]] = {}
    for sym, dates in earnings_map.items():
        best: dict[str, Any] | None = None
        for d in dates:
            bdays = int(np.busday_count(ref_day, d)) if d >= ref_day else -int(np.busday_count(d, ref_day))
            if -earnings_window_days <= bdays <= earnings_window_days:
                if best is None or abs(bdays) < abs(best["bdays"]):
                    status = "EARNINGS_SOON" if bdays >= 0 else "POST_EARNINGS_REPRICING"
                    best = {"status": status, "date": str(d), "bdays": bdays}
        if best:
            earnings_flags[sym] = best

    return {
        "as_of": str(ref_day),
        "macro_alerts": list(dict.fromkeys(macro_alerts)),
        "earnings_flags": earnings_flags,
        "generated_at": now_dt.isoformat(timespec="seconds"),
    }


_SESSION_TO_SOURCE = {
    "OVN": "overnight",
    "PRE": "pre_market",
    "RTH": "regular",
    "POST": "post_market",
}


def load_webull_live_quotes(
    settings: Settings,
    symbols: list[str],
) -> dict[str, dict[str, Any]]:
    """通过 Webull SDK 拉取 24 小时最新 1 分钟 K 线（支持夜盘 OVN / 盘前 PRE / 常规 RTH / 盘后 POST）。

    只读行情接口，不涉及账户或下单；未配置密钥或网络不可用时自动返回空字典。
    """
    if not settings.has_webull_credentials:
        return {}
    syms = list(dict.fromkeys(s.upper() for s in symbols))
    if not syms:
        return {}
    try:
        from .intraday_data import SESSIONS, _to_frame
        from .webull_client import WebullClient, _json, _unwrap_list

        client = WebullClient(settings)
        out: dict[str, dict[str, Any]] = {}
        for i in range(0, len(syms), 20):
            batch = syms[i : i + 20]
            raw = _json(
                client.data.market_data.get_batch_history_bar(
                    batch, "US_STOCK", "M1", count="3", trading_sessions=SESSIONS
                )
            )
            for item in _unwrap_list(raw):
                sym = str(item.get("symbol") or "").upper()
                rows = item.get("result") or []
                if not sym or not rows:
                    continue
                df = _to_frame(rows)
                if df.empty:
                    continue
                last_price = float(df["close"].iloc[-1])
                if not np.isfinite(last_price) or last_price <= 0:
                    continue
                session = str(df["session"].iloc[-1]).upper()
                ts = df.index[-1]
                out[sym] = {
                    "price": last_price,
                    "session": session,
                    "source": _SESSION_TO_SOURCE.get(session, "live"),
                    "timestamp_et": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                }
        return out
    except Exception as e:
        log.warning("拉取 Webull 24 小时实时报价失败，回退 Yahoo 报价: %s", e)
        return {}


def fetch_kospi_move(session: requests.Session | None = None) -> float | None:
    """获取韩国 KOSPI 指数当天相对上一交易日收盘的涨跌幅（小数，例如 -0.05 表示 -5.0%）。"""
    sess, crumb = (session, "") if session is not None else _make_yahoo_session()
    sess.headers.setdefault("User-Agent", USER_AGENT)
    params: dict[str, Any] = {"interval": "1d", "range": "10d"}
    if crumb:
        params["crumb"] = crumb
    for attempt in range(2):
        try:
            r = sess.get(YAHOO_CHART_URL.format("%5EKS11"), params=params, timeout=12)
            if r.status_code != 200:
                time.sleep(1.0)
                continue
            res = (r.json().get("chart") or {}).get("result") or []
            if not res:
                return None
            node = res[0]
            meta = node.get("meta") or {}
            closes = [float(c) for c in (node.get("indicators", {}).get("quote", [{}])[0].get("close") or []) if c is not None]
            reg = _raw(meta.get("regularMarketPrice"))
            prev = _raw(meta.get("chartPreviousClose")) or _raw(meta.get("previousClose"))
            if len(closes) >= 2 and closes[-2] > 0:
                cur = reg if (reg and reg > 0) else closes[-1]
                return float(cur / closes[-2] - 1.0)
            if reg and prev and prev > 0:
                return float(reg / prev - 1.0)
        except Exception as e:
            log.debug("获取 KOSPI 涨跌第 %d 次失败: %s", attempt + 1, e)
            time.sleep(1.0)
    return None


def load_regime_context(
    bars: dict[str, pd.DataFrame],
    live_quotes: dict[str, dict[str, Any]] | None = None,
    refresh_kospi: bool = True,
) -> dict[str, Any]:
    """计算与 swing 系统同口径的市场趋势与跨市场冲击状态（SOXX/QQQ 三票制 + KOSPI + SOXL 夜盘）。"""
    from .swing import ensemble_votes, trend_regime

    ctx: dict[str, Any] = {
        "soxx_votes": None,
        "soxx_regime": None,
        "soxx_date": None,
        "qqq_votes": None,
        "qqq_regime": None,
        "qqq_date": None,
        "kospi_move_pct": None,
        "soxl_move_pct": None,
        "summary": "",
    }

    for key, sym in (("soxx", "SOXX"), ("qqq", "QQQ")):
        df = bars.get(sym)
        if df is not None and len(df) >= 60:
            v = int(ensemble_votes(df).iloc[-1])
            ctx[f"{key}_votes"] = v
            ctx[f"{key}_regime"] = int(trend_regime(v))
            ctx[f"{key}_date"] = str(df.index[-1].date())

    if refresh_kospi:
        km = fetch_kospi_move()
        if km is not None and np.isfinite(km):
            ctx["kospi_move_pct"] = round(km * 100.0, 2)

    if live_quotes and "SOXL" in live_quotes and "SOXL" in bars and len(bars["SOXL"]) > 0:
        last_c = float(bars["SOXL"]["close"].iloc[-1])
        cur_p = float(live_quotes["SOXL"]["price"])
        if last_c > 0:
            ctx["soxl_move_pct"] = round((cur_p / last_c - 1.0) * 100.0, 2)

    parts: list[str] = []
    label_map = {1: "上涨趋势", 0: "震荡过渡", -1: "下跌趋势"}
    if ctx["soxx_votes"] is not None:
        parts.append(f"SOXX 三票={ctx['soxx_votes']:+d}({label_map[ctx['soxx_regime']]})")
    if ctx["qqq_votes"] is not None:
        parts.append(f"QQQ 三票={ctx['qqq_votes']:+d}({label_map[ctx['qqq_regime']]})")
    if ctx["kospi_move_pct"] is not None:
        parts.append(f"KOSPI {ctx['kospi_move_pct']:+.2f}%")
    if ctx["soxl_move_pct"] is not None:
        parts.append(f"SOXL 夜盘/实时 {ctx['soxl_move_pct']:+.2f}%")
    ctx["summary"] = " | ".join(parts)
    return ctx
