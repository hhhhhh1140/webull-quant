"""半导体波段系统的实盘 / 沙盒运行。两层，各自记账，互不干扰：

核心层（core）：每天一次。三条只做多趋势规则（SOXX 日线）的组合权重 → SOXL 目标仓位，偏差超过阈值才调仓。
战术层（tactical）：按美股交易日运行，核心仓监控覆盖夜盘至盘后。
    04:00 ET（夜盘结束）看 SOXL 相对昨收涨跌 + 韩国（EWY）确认 → 顺势买 SOXL 或 SOXS
    软件止损（持有标的跌 stop_pct），止损后从最低点反弹 rebound_pct 买回一次
    15:55 ET 全部平仓，不留隔夜

两层的持仓记在 state/swing_books_<env>.json，和选股系统的账本分开。
"""
from __future__ import annotations

import json
import math
import time as _time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Callable, Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from . import journal
from .config import ROOT, Settings
from .risk import KILL_SWITCH, OrderIntent
from .swing import ensemble_votes, ensemble_weight, trend_regime

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------- 配置
DEFAULTS = {
    "capital": 100_000,
    "market_source": "webull",   # webull | futu | auto（futu 统一使用富途 OpenD 24h 1m/5m/日线，实盘免 Webull 夜盘订阅）
    "core": {"enabled": True, "index": "SOXX", "long": "SOXL", "allocation": 0.30, "rebalance_band": 0.05},
    "tactical": {"enabled": True, "long": "SOXL", "short": "SOXS",
                 "checkpoints": ["21:00", "23:00", "01:30", "04:00"],   # 美东；21:00/23:00 属于前一天晚上
                 "exit": "15:55", "threshold": 0.03,
                 "korea": "KOSPI", "korea_confirm": 0.01,               # KOSPI 综合指数当天同方向至少 1%
                 "capital": 3000, "position_frac": 0.33,               # 系统本金（独立记账）和每笔仓位比例
                 "stop_pct": 0.06, "reentry": True, "rebound_pct": 0.03, "last_reentry": "15:00",
                 "max_reentries": None,                                  # 一天买回次数不限
                 "reentry_stop_pct": None,                               # 买回后第 2 腿兜底止损（None = 沿用 stop_pct/trail_pct）
                 "reentry_position_frac": None,                          # None = 沿用首腿的仓位比例
                 "reentry_trail_pct": None,                              # 从本次买回后的最高价计算
                 # 三票制双层动态切换：上涨趋势拿核心 SOXL 过夜（从当日最高价回撤 12% 止损、次日起较上一交易日常规盘收盘价上涨 5% 买回）；
                 # 非上涨趋势清空核心、切入战术层（下跌趋势只做空 SOXS，震荡期双向）
                 "trend_switch": {"enabled": False, "index": "SOXX", "core_symbol": "SOXL",
                                  "core_frac": 1.0, "trail_pct": 0.12, "rebuy_pct": 0.05,
                                  "core_session_start": "09:30", "rebuy_session": "ALL"},
                 # 融资与盈利加仓：夜盘不能融资（最多 100%），盘前 / 常规时段最多 130%
                 "max_frac_ovn": 1.0, "max_frac": 1.3,
                 "add_step": 0.03, "add_frac": 0.33, "add_from": "04:00", "add_last": "15:00",
                 "trail_after_add": 0.06,
                 "trail_pct": None,                                      # 移动止损：从持仓最高价回撤这个比例就卖（之后反弹 rebound_pct 买回）
                 "poll_seconds": 60, "sides": ["long", "short"],
                 # 分时路径：不只看"现在相对昨收"，还看昨收以来（盘后 + 夜盘 + 盘前）的高低点和 VWAP
                 "path": {"enabled": False,
                          "vwap_filter": True,      # 做多要在 VWAP 上方，做空要在 VWAP 下方
                          "veto_pct": None,         # 做多但已经从高点回落 ≥ 这个比例 / 做空但已经从低点反弹 ≥ 这个比例 → 不做
                          "reversal_pct": None,     # 从低点反弹（或从高点回落）≥ 这个比例，算作反转信号
                          "reversal_min_move": 0.0, # 反转信号还要求相对昨收至少翻到这个涨跌（做多 ≥ +x，做空 ≤ -x）
                          "reversal_korea": True,   # 反转信号是否也要韩国确认
                          "scan": "checkpoints",
                          "scan_from": None, "scan_to": None},  # 连续扫描的起止时间（美东），None = 第一个 / 最后一个检查时点   # checkpoints = 只在检查时点判断；continuous = 首个到最后一个时点之间每次轮询都判断
                 # 宏观利率门控：开启后要求全部利率标的（如 ^IRX、^TNX）昨日收盘均高于 SMA(window)，才允许夜盘做空 SOXS
                 "macro_short_gate": {"enabled": False, "symbols": ["^IRX", "^TNX"], "sma_window": 60},
                 # 动态仓位 = base × 信号强弱倍数 × 趋势效率系数，限制在 [min, max]
                 "sizing": {"mode": "dynamic", "base": 0.33, "signal_cap": 2.0, "use_trend_efficiency": False,
                            "trend_index": "SOXX", "er_window": 10, "er_tiers": [[0.2, 0.6], [0.4, 1.0], [99, 1.3]],
                            "min": 0.15, "max": 0.80}},
    "order": {"type": "LIMIT", "limit_offset_bps": 30, "time_in_force": "DAY", "trading_session": "ALL_DAY"},
}


def _deep_merge(base: dict, over: dict) -> dict:
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def swing_cfg(settings: Settings) -> dict:
    """默认值 + config.yaml 的 swing 部分（逐层合并，只写部分字段也不会丢掉其他默认值）。"""
    overrides = settings.section("swing")
    cfg = _deep_merge(json.loads(json.dumps(DEFAULTS)), overrides)
    old = (overrides.get("tactical") or {}).get("trend_switch") or {}
    if "trail_pct" not in old:
        previous_pct = old.get("stop_pct", old.get("day_stop_pct"))
        if previous_pct is not None:
            cfg["tactical"]["trend_switch"]["trail_pct"] = previous_pct
    return cfg


# ---------------------------------------------------------------- 账本
class Books:
    def __init__(self, env: str):
        self.path = journal.STATE / f"swing_books_{env}.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"core": {}, "tactical": {}}

    def qty(self, book: str, sym: str) -> float:
        return float(self.data.setdefault(book, {}).get(sym, 0))

    def add(self, book: str, sym: str, dq: float) -> None:
        b = self.data.setdefault(book, {})
        b[sym] = round(float(b.get(sym, 0)) + dq, 6)
        if abs(b[sym]) < 1e-9:
            b.pop(sym)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


# ---------------------------------------------------------------- 行情接口（方便测试时替换）
class Market(Protocol):
    def last_price(self, sym: str) -> tuple[datetime, float]: ...
    def minute_quote(self, sym: str, since: datetime | None, now: datetime) -> tuple[datetime, float, float] | None: ...
    def prev_rth_close(self, sym: str, day: date) -> float: ...
    def daily(self, sym: str, count: int) -> pd.DataFrame: ...
    def kospi_move(self, now: datetime) -> float | None: ...
    def path_stats(self, sym: str, day: date) -> tuple[float, float, float | None]: ...


def _kospi_daily_closes() -> list[tuple[date, float]]:
    """KOSPI 最近几天的官方收盘价（Yahoo 日线，含收盘集合竞价），[(韩国日期, 收盘)]；取不到返回空。
    当天还没收盘时，最后一条是盘中实时价。"""
    import requests
    from .intraday_data import KOSPI_URL
    try:
        r = requests.get(KOSPI_URL, params={"interval": "1d", "range": "10d"},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        j = r.json()["chart"]["result"][0]
        days = pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert("Asia/Seoul").date
        return [(d, float(c)) for d, c in zip(days, j["indicators"]["quote"][0]["close"]) if c is not None]
    except Exception:
        return []


def _kospi_prev_close(kday: date) -> float | None:
    """KOSPI 在 kday（韩国日期）之前最后一个交易日的官方收盘价。取不到返回 None。"""
    closes = [c for d, c in _kospi_daily_closes() if d < kday]
    return closes[-1] if closes else None


_MACRO_GATE_CACHE: dict[tuple, tuple[bool, str]] = {}


def fetch_yahoo_daily_closes(symbol: str, range_: str = "1y") -> pd.Series:
    """拉取 Yahoo Finance 日线收盘价（美东交易日索引），供宏观利率门控使用。"""
    from urllib.parse import quote
    import requests
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
    last_exc: Exception | None = None
    for _ in range(2):
        try:
            r = requests.get(url, params={"interval": "1d", "range": range_},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
            r.raise_for_status()
            j = r.json()["chart"]["result"][0]
            days = pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert("America/New_York").date
            closes = j["indicators"]["quote"][0]["close"]
            s = pd.Series(closes, index=days, dtype=float).dropna()
            return s[~s.index.duplicated(keep="last")].sort_index()
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"无法拉取 {symbol} 日线：{last_exc}")


def eval_macro_short_gate(day: date, gate_cfg: dict, fetcher=fetch_yahoo_daily_closes) -> tuple[bool, str]:
    """评估宏观利率紧缩门控（仅用 < day 已完成美东日线，无未来函数）。
    当全部 symbols 的最新收盘价 > SMA(sma_window) 时返回 True，否则禁止夜盘做空 SOXS。"""
    if not (gate_cfg or {}).get("enabled"):
        return True, ""
    symbols = tuple(gate_cfg.get("symbols") or ("^IRX", "^TNX"))
    window = int(gate_cfg.get("sma_window") or 60)
    use_cache = fetcher is fetch_yahoo_daily_closes
    cache_key = (day, symbols, window)
    if use_cache and cache_key in _MACRO_GATE_CACHE:
        return _MACRO_GATE_CACHE[cache_key]
    parts = []
    all_up = True
    try:
        for sym in symbols:
            s = fetcher(sym)
            s = s[s.index < day]
            if len(s) < window:
                res = (False, f"利率门控：{sym} 有效日线不足 {window} 根 → 暂禁夜盘做空 SOXS")
                return res
            last_val = float(s.iloc[-1])
            sma_val = float(s.iloc[-window:].mean())
            ok = last_val > sma_val
            if not ok:
                all_up = False
            parts.append(f"{sym} {last_val:.2f}{'>' if ok else '<='}SMA{window}({sma_val:.2f})")
    except Exception as exc:
        return False, f"利率门控拉取失败（{exc}）→ 保守禁夜盘做空 SOXS"
    verdict = "允许夜盘做空 SOXS" if all_up else "未双紧缩，禁夜盘做空 SOXS"
    res = (all_up, f"利率门控：{' 且 '.join(parts)} → {verdict}")
    if use_cache:
        _MACRO_GATE_CACHE[cache_key] = res
    return res


KOSPI_CLOSE_KST = time(15, 35)   # 15:30 收盘，留 5 分钟让官方收盘价出来


def fetch_kospi_move(now: datetime) -> float | None:
    """KOSPI 当天涨跌（相对上一交易日收盘），用 Yahoo 5 分钟线，只取已收盘的 K 线。"""
    import requests
    from .intraday_data import KOSPI_URL
    r = requests.get(KOSPI_URL, params={"interval": "5m", "range": "5d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    j = r.json()["chart"]["result"][0]
    ts = pd.to_datetime(j["timestamp"], unit="s", utc=True) + pd.Timedelta(minutes=5)
    df = pd.DataFrame({"close": j["indicators"]["quote"][0]["close"]}, index=ts).dropna()
    df = df[df.index <= pd.Timestamp(now).tz_convert("UTC")]
    if df.empty:
        return None
    kday = (df.index.tz_convert("Asia/Seoul") - pd.Timedelta(minutes=5)).date
    last_day = kday[-1]
    today_kst = pd.Timestamp(now).tz_convert("Asia/Seoul").date()
    if last_day != today_kst:
        return None                      # 今天韩国休市或还没开盘：不拿昨天的数据冒充
    daily = _kospi_daily_closes()
    prevs = [c for d, c in daily if d < last_day]
    prev_close = prevs[-1] if prevs else None
    if prev_close is None:
        prev = df[kday < last_day]
        if prev.empty:
            return None
        prev_close = float(prev.close.iloc[-1])
    cur = float(df.close.iloc[-1])
    if pd.Timestamp(now).tz_convert("Asia/Seoul").time() >= KOSPI_CLOSE_KST:
        today = [c for d, c in daily if d == last_day]
        if today:
            cur = today[-1]
    return float(cur / prev_close - 1)


def make_market(settings: Settings, now: Callable[[], datetime], *, source: str | None = None, client=None) -> Market:
    """按配置创建实时行情源（futu / webull / auto）。"""
    chosen = (source or swing_cfg(settings).get("market_source") or "webull").strip().lower()
    if chosen == "futu":
        from .futu_gateway import FutuMarket
        return FutuMarket(now)
    if chosen == "auto":
        try:
            from .futu_gateway import FutuMarket
            fm = FutuMarket(now)
            fm.last_price(swing_cfg(settings)["tactical"]["long"])
            return fm
        except Exception as exc:
            print(f"[market] 富途行情不可用（{exc}），回退到 Webull 行情")
    if client is None:
        from .webull_client import WebullClient
        for attempt in range(1, 11):
            try:
                client = WebullClient(settings)
                break
            except Exception as e:
                if attempt == 10:
                    raise
                print(f"[swing] 连接 Webull 失败（第 {attempt}/10 次），10 秒后重试：{e}")
                _time.sleep(10)
    return WebullMarket(client, now)


class WebullMarket:
    def __init__(self, client, now: Callable[[], datetime]):
        self.c, self.now = client, now

    def _bars(self, sym: str, ts: str, count: int, **kwargs) -> pd.DataFrame:
        from .intraday_data import SESSIONS, _to_frame
        from .webull_client import _json, _unwrap_list
        d = _unwrap_list(_json(self.c.data.market_data.get_batch_history_bar(
            [sym], "US_STOCK", ts, count=str(count), trading_sessions=SESSIONS, **kwargs)))
        return _to_frame(d[0]["result"]) if d else pd.DataFrame()

    def last_price(self, sym: str) -> tuple[datetime, float]:
        b = self._bars(sym, "M1", 3)
        return b.index[-1].to_pydatetime(), float(b.close.iloc[-1])

    def minute_quote(self, sym: str, since: datetime | None, now: datetime) -> tuple[datetime, float, float] | None:
        """返回新收盘的分钟线中最后一根的结束时刻、收盘价和这段的最高价。"""
        minute = timedelta(minutes=1)
        count = 3 if since is None else min(1200, max(3, math.ceil((now - since) / minute) + 1))
        bars = self._bars(sym, "M1", count)
        if bars.empty:
            return None
        # A full session can exceed the Webull page limit after a late start.
        if since is not None and len(bars) == 1200 and bars.index[0] > since:
            end_time = int((bars.index[0] - timedelta(seconds=1)).timestamp() * 1000)
            remaining = max(3, math.ceil((bars.index[0] - since) / minute) + 1)
            earlier = self._bars(sym, "M1", min(1200, remaining), end_time=end_time)
            if not earlier.empty:
                bars = pd.concat([earlier.loc[earlier.index < bars.index[0]], bars])
        bars = bars.loc[bars.index + minute <= now]
        bars = bars.iloc[-1:] if since is None else bars.loc[bars.index >= since]
        if bars.empty:
            return None
        return (bars.index[-1].to_pydatetime() + minute, float(bars.close.iloc[-1]), float(bars.high.max()))

    def prev_rth_close(self, sym: str, day: date) -> float:
        b = self._bars(sym, "M5", 600)
        rth = b[(b.session == "RTH") & (b.index.date < day)]
        return float(rth.close.iloc[-1])

    def path_stats(self, sym: str, day: date) -> tuple[float, float, float | None]:
        """昨天常规时段收盘以来（盘后 + 夜盘 + 盘前 + 今天）的最低、最高和 VWAP，用 5 分钟线。"""
        b = self._bars(sym, "M5", 600)
        rth = b[(b.session == "RTH") & (b.index.date < day)]
        w = b[b.index > rth.index[-1]]
        if w.empty:
            return float("nan"), float("nan"), None
        vol = w.volume.sum()
        vwap = float(((w.high + w.low + w.close) / 3 * w.volume).sum() / vol) if vol > 0 else None
        return float(w.low.min()), float(w.high.max()), vwap

    def kospi_move(self, now: datetime) -> float | None:
        return fetch_kospi_move(now)

    def daily(self, sym: str, count: int) -> pd.DataFrame:
        from .webull_client import _json, _unwrap_list, bars_to_frame
        d = _unwrap_list(_json(self.c.data.market_data.get_batch_history_bar([sym], "US_STOCK", "D", count=str(count))))
        df = bars_to_frame(d[0]["result"])
        now = self.now()
        if now.time() < time(16, 5):          # 今天还没收盘：去掉今天这根未完成的日线
            df = df[df.index.date < now.date()]
        return df

    def macro_short_allowed(self, day: date, gate_cfg: dict) -> tuple[bool, str]:
        return eval_macro_short_gate(day, gate_cfg)


class CachedMarket:
    """短时行情缓存包装器：让多套并行模拟盘在同一轮询内共享 Webull/Yahoo 行情，避免重复请求。"""

    def __init__(self, inner: Market, quote_ttl: float = 15.0, slow_ttl: float = 300.0):
        self.inner = inner
        self.quote_ttl = quote_ttl
        self.slow_ttl = slow_ttl
        self._cache: dict[tuple, tuple[float, object]] = {}

    def _get(self, key: tuple, ttl: float, fn):
        now_t = _time.monotonic()
        hit = self._cache.get(key)
        if hit is not None and now_t - hit[0] <= ttl:
            return hit[1]
        val = fn()
        self._cache[key] = (now_t, val)
        return val

    def last_price(self, sym: str) -> tuple[datetime, float]:
        return self._get(("last", sym), self.quote_ttl, lambda: self.inner.last_price(sym))

    def minute_quote(self, sym: str, since: datetime | None, now: datetime) -> tuple[datetime, float, float] | None:
        key = ("mq", sym, since.isoformat() if since else None, now.replace(second=0, microsecond=0).isoformat())
        return self._get(key, self.quote_ttl, lambda: self.inner.minute_quote(sym, since, now))

    def prev_rth_close(self, sym: str, day: date) -> float:
        return self._get(("prev", sym, day), self.slow_ttl, lambda: self.inner.prev_rth_close(sym, day))

    def daily(self, sym: str, count: int) -> pd.DataFrame:
        return self._get(("daily", sym, count), self.slow_ttl, lambda: self.inner.daily(sym, count))

    def kospi_move(self, now: datetime) -> float | None:
        key = ("kospi", now.replace(second=0, microsecond=0).isoformat())
        return self._get(key, self.quote_ttl, lambda: self.inner.kospi_move(now))

    def path_stats(self, sym: str, day: date) -> tuple[float, float, float | None]:
        return self._get(("path", sym, day), self.quote_ttl, lambda: self.inner.path_stats(sym, day))

    def macro_short_allowed(self, day: date, gate_cfg: dict) -> tuple[bool, str]:
        syms = tuple((gate_cfg or {}).get("symbols") or ())
        win = int((gate_cfg or {}).get("sma_window") or 60)
        key = ("macro", day, bool((gate_cfg or {}).get("enabled")), syms, win)
        return self._get(key, self.slow_ttl, lambda: eval_macro_short_gate(day, gate_cfg)
                         if not hasattr(self.inner, "macro_short_allowed")
                         else self.inner.macro_short_allowed(day, gate_cfg))


# ---------------------------------------------------------------- 下单
def _submit(broker, sym: str, side: str, qty: int, price: float, reason: str, *, stop_price=None) -> bool:
    if qty <= 0:
        return False
    intent = OrderIntent(sym, side, qty, price, reason)
    try:
        res = broker.submit(intent)
    except Exception as e:
        res = getattr(e, "order_attempt", {"status": "error"})
        print(f"  下单失败: {e}")
        error = str(e)
    else:
        error = None
    from .wechat import record_order

    try:
        record_order(broker, intent, res, stop_price=stop_price)
    except Exception as exc:
        print(f"  微信通知写入失败（{type(exc).__name__}），交易按原订单结果处理")
    ok = res.get("status") not in ("error", "rejected")
    journal.record("swing_order", symbol=sym, side=side, qty=qty, price=price, reason=reason,
                   status=res.get("status"), order=res.get("order"), stop_price=stop_price, error=error)
    return ok


def make_broker(settings: Settings, mode: str, confirm_live: bool, order_cfg: dict, *, label: str = "paper"):
    from .broker import DryRunBroker, LocalSimBroker, WebullBroker
    if mode == "webull":
        from .webull_client import WebullClient
        return WebullBroker(settings, WebullClient(settings), order_cfg, confirm_live=confirm_live)
    if mode == "paper":
        return LocalSimBroker(order_cfg, label=label)
    if mode == "futu-sim":
        from .futu_gateway import FutuSimBroker
        return FutuSimBroker(order_cfg, label=label)
    return DryRunBroker(order_cfg)


# ---------------------------------------------------------------- 核心层
def core_rebalance(settings: Settings, md: Market, broker, books: Books, dry: bool = False) -> dict:
    cfg = swing_cfg(settings)
    c = cfg["core"]
    if KILL_SWITCH.exists():
        print("KILL_SWITCH 存在，核心层不调仓")
        return {"skipped": "kill_switch"}
    X = md.daily(c["index"], 300)
    w = float(ensemble_weight(X).iloc[-1])
    _, px = md.last_price(c["long"])
    budget = cfg["capital"] * c["allocation"]
    target = math.floor(budget * w / px)
    cur = int(books.qty("core", c["long"]))
    diff = target - cur
    info = {"index_date": str(X.index[-1].date()), "weight": round(w, 3), "price": px, "target_qty": target,
            "current_qty": cur, "diff": diff}
    print(f"[核心层] {c['index']} 截至 {info['index_date']}：{int(round(w * 3))}/3 条规则看多 → 目标 {c['long']} {target} 股"
          f"（当前 {cur} 股，约 ${target * px:,.0f} / 预算 ${budget:,.0f}）")
    big_enough = abs(diff) * px >= c["rebalance_band"] * budget or (target == 0 and cur > 0)
    if diff and big_enough:
        side = "BUY" if diff > 0 else "SELL"
        if _submit(broker, c["long"], side, abs(diff), px, f"核心层调仓 权重 {w:.2f}") and not dry:
            books.add("core", c["long"], diff)
        info["order"] = f"{side} {abs(diff)}"
    else:
        print("  变化不大，不调仓")
    journal.record("swing_core", **info)
    return info


# ---------------------------------------------------------------- 战术层
@dataclass
class TacState:
    day: str
    evaluated: list = field(default_factory=list)   # 已经检查过的时点
    decided: bool = False                           # 已经进过场（或全部时点都检查完了）
    side: int = 0
    symbol: str = ""
    move: float | None = None
    korea_move: float | None = None
    gate: str = ""                                  # 三票制方向限制
    frac: float = 0.0                               # 当天的仓位比例（止损后买回沿用）
    sizing_note: str = ""
    qty: int = 0
    entry: float = 0.0                              # 持仓均价（加仓后更新）
    frac_total: float = 0.0                         # 当前持仓占权益的比例
    last_add: float = 0.0                           # 上一次买入价（加仓门槛按它算）
    adds: int = 0
    peak: float = 0.0
    trades: int = 0
    stopped: bool = False
    low_after_stop: float | None = None
    reentries: int = 0
    reentry_checked_at: str = ""
    scan_note: str = ""                             # 连续扫描上一次的结论（只在变化时写日志）
    done: bool = False
    pnl: float = 0.0
    equity_start: float = 0.0
    trend_votes: int | None = None                  # 三票制投票总分（-3 ~ +3）
    trend_regime: int | None = None                 # +1 上涨 / 0 震荡 / -1 下跌
    trend_note: str = ""
    core_active_today: bool = False                 # 今天是否由核心层接管
    core_prev_close: float | None = None
    macro_short_allowed: bool | None = None         # 宏观利率门控结果（True 允许夜盘做空 SOXS / False 禁止）
    macro_short_note: str = ""
    log: list = field(default_factory=list)


@dataclass
class CoreState:
    """持仓可以跨日，高点只在对应的美股交易日内有效。"""
    holding: bool = False
    entry: float = 0.0
    peak: float = 0.0
    peak_day: str = ""
    peak_checked_at: str = ""
    stopped: bool = False
    stopped_day: str = ""
    last_votes: int | None = None
    last_regime: int | None = None
    last_index_date: str = ""


def _core_state_path(env: str):
    return journal.STATE / f"swing_core_state_{env}.json"


def _load_core_state(env: str) -> CoreState:
    p = _core_state_path(env)
    if p.exists():
        raw = json.loads(p.read_text())
        valid = {f.name for f in CoreState.__dataclass_fields__.values()}
        return CoreState(**{k: v for k, v in raw.items() if k in valid})
    return CoreState()


def _save_core_state(env: str, cst: CoreState) -> None:
    p = _core_state_path(env)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(cst), ensure_ascii=False, indent=1))


def _state_path(env: str, day: date):
    return journal.STATE / f"swing_tactical_{env}_{day}.json"


def _load_state(env: str, day: date) -> TacState:
    p = _state_path(env, day)
    if p.exists():
        raw = json.loads(p.read_text())
        valid = {f.name for f in TacState.__dataclass_fields__.values()}
        return TacState(**{k: v for k, v in raw.items() if k in valid})
    return TacState(day=str(day))


def _save_state(env: str, st: TacState) -> None:
    p = _state_path(env, date.fromisoformat(st.day))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(st), ensure_ascii=False, indent=1))


def _at(day: date, hhmm: str) -> datetime:
    """交易日 day 的美东时间点；16:00 以后的时间属于前一天晚上（夜盘）。"""
    h, m = map(int, hhmm.split(":"))
    d = day - timedelta(days=1) if h >= 16 else day
    if h >= 16:                       # 周一的夜盘从周日晚上开始
        while d.weekday() == 5:
            d -= timedelta(days=1)
    return datetime.combine(d, time(h, m), tzinfo=ET)


# ---- 系统权益（独立于券商账户记账：本金 + 已实现盈亏）
def _equity_path(env: str):
    return journal.STATE / f"swing_equity_{env}.json"


def system_equity(settings: Settings) -> float:
    p = _equity_path(settings.environment)
    cap = float(swing_cfg(settings)["tactical"]["capital"])
    if p.exists():
        return float(json.loads(p.read_text()).get("equity", cap))
    return cap


def _add_pnl(settings: Settings, pnl: float, day: str) -> None:
    p = _equity_path(settings.environment)
    cap = float(swing_cfg(settings)["tactical"]["capital"])
    d = json.loads(p.read_text()) if p.exists() else {"capital": cap, "equity": cap, "history": []}
    d["equity"] = round(d["equity"] + pnl, 2)
    d["history"].append({"day": day, "pnl": round(pnl, 2), "equity": d["equity"]})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1))


def tactical_directions(config: dict, regime: int | None) -> tuple[set[str], str]:
    sides = set(config["sides"])
    if regime == 1:
        sides &= {"long"}
    elif regime == -1:
        sides &= {"short"}
    description = {
        frozenset({"long", "short"}): "规则允许双向",
        frozenset({"long"}): "规则只允许买 SOXL",
        frozenset({"short"}): "规则只允许买 SOXS",
    }
    return sides, description.get(frozenset(sides), "规则禁开仓")


def efficiency_ratio(close: pd.Series, n: int) -> float | None:
    """趋势效率：n 天净变化 / 每天变化绝对值之和。接近 1 = 单边趋势，接近 0 = 来回震荡。"""
    if len(close) < n + 1:
        return None
    seg = close.iloc[-(n + 1):]
    path = seg.diff().abs().sum()
    return float(abs(seg.iloc[-1] - seg.iloc[0]) / path) if path > 0 else None


def position_frac(t: dict, move: float, korea_move: float | None, er: float | None) -> tuple[float, str]:
    """返回 (仓位比例, 说明)。固定模式直接用 position_frac。"""
    sz = t.get("sizing") or {}
    if sz.get("mode") != "dynamic":
        return t["position_frac"], f"固定 {t['position_frac']:.0%}"
    score = abs(move) / t["threshold"]
    if korea_move is not None and t.get("korea_confirm"):
        score = min(score, abs(korea_move) / t["korea_confirm"])
    sig = min(max(score, 1.0), sz.get("signal_cap", 2.0))
    ef = 1.0
    if er is not None and sz.get("use_trend_efficiency", False):
        for upper, factor in sz.get("er_tiers", []):
            if er < upper:
                ef = factor
                break
    frac = min(max(sz.get("base", 0.33) * sig * ef, sz.get("min", 0.15)), sz.get("max", 0.8))
    if not sz.get("use_trend_efficiency", False):
        return frac, f"仓位 {frac:.0%} = {sz.get('base', 0.33):.0%} × 信号强弱 {sig:.2f}"
    er_txt = f"{er:.2f}" if er is not None else "无数据"
    return frac, (f"仓位 {frac:.0%} = {sz.get('base', 0.33):.0%} × 信号 {sig:.2f} × 趋势效率系数 {ef:.1f}"
                  f"（{sz.get('trend_index', 'SOXX')} {sz.get('er_window', 10)} 日趋势效率 {er_txt}）")


def _continuous(t: dict) -> bool:
    pc = t.get("path") or {}
    return bool(pc.get("enabled")) and pc.get("scan") == "continuous"


def _scan_window(t: dict, day: date) -> tuple[datetime, datetime]:
    pc = t.get("path") or {}
    return _at(day, pc.get("scan_from") or t["checkpoints"][0]), _at(day, pc.get("scan_to") or t["checkpoints"][-1])


def anchor_sides(move: float, from_low: float, from_high: float, th: float, anchor: str) -> dict:
    """按锚点给出可做的方向和强度 {+1: 强度, -1: 强度}。
    prev_close = 相对昨收；extreme = 相对昨收以来的最低 / 最高（从 -3% 拉到 0 也算涨 3%）；either = 两者任一。"""
    out = {}
    up = [move] if anchor in ("prev_close", "either") else []
    dn = [move] if anchor in ("prev_close", "either") else []
    if anchor in ("extreme", "either"):
        up.append(from_low)
        dn.append(from_high)
    if up and max(up) >= th:
        out[1] = max(up)
    if dn and min(dn) <= -th:
        out[-1] = -min(dn)
    return out


def pick_side(cands: dict, low_is_later: bool) -> int:
    """两个方向都满足（先大涨后大跌或反过来）时，按最近一个极值判断：最近是低点 → 正在反弹 → 做多。"""
    if len(cands) == 2:
        return 1 if low_is_later else -1
    return next(iter(cands), 0)


def path_signal(move: float, px: float, lo: float, hi: float, vwap: float | None, threshold: float,
                pc: dict) -> tuple[int, str, bool]:
    """根据昨收以来的分时路径给方向。返回 (方向 +1/-1/0, 说明, 是否反转信号)。
    move = 现价相对昨收；lo / hi = 昨收以来最低 / 最高；vwap = 昨收以来成交量加权均价。"""
    side = 1 if move >= threshold else -1 if move <= -threshold else 0
    if not pc.get("enabled"):
        return side, "", False
    from_low, from_high = px / lo - 1, px / hi - 1
    desc = f"昨收以来低 {lo / px * (1 + move) - 1:+.1%} / 高 {hi / px * (1 + move) - 1:+.1%}，从低点 {from_low:+.1%}、从高点 {from_high:+.1%}"
    if vwap:
        desc += f"，VWAP {vwap:.2f}（现价{'上' if px >= vwap else '下'}方）"
    veto, rev = pc.get("veto_pct"), pc.get("reversal_pct")
    use_vwap = pc.get("vwap_filter") and vwap
    if side > 0:
        if use_vwap and px < vwap:
            return 0, desc + " → 涨了但在 VWAP 下方，不追", False
        if veto and from_high <= -veto:
            return 0, desc + f" → 已经从高点回落 {-from_high:.1%}，不追", False
    elif side < 0:
        if use_vwap and px > vwap:
            return 0, desc + " → 跌了但在 VWAP 上方，不空", False
        if veto and from_low >= veto:
            return 0, desc + f" → 已经从低点反弹 {from_low:.1%}，不空", False
    elif rev:
        m = pc.get("reversal_min_move", 0.0) or 0.0
        if from_low >= rev and move >= m and (not use_vwap or px >= vwap):
            return 1, desc + f" → 从低点反弹 {from_low:.1%}，V 形反转做多", True
        if from_high <= -rev and move <= -m and (not use_vwap or px <= vwap):
            return -1, desc + f" → 从高点回落 {-from_high:.1%}，倒 V 反转做空", True
    return side, desc, False


def eval_trend_switch(settings: Settings, md: Market, day: date) -> tuple[int, int, str]:
    """用 1 倍指数（SOXX）日线计算三票制总分（-3~+3）和趋势状态（+1 上涨 / 0 震荡 / -1 下跌）。"""
    ts = swing_cfg(settings)["tactical"].get("trend_switch") or {}
    idx_sym = ts.get("index", "SOXX")
    daily = md.daily(idx_sym, 120)
    daily = daily.loc[daily.index.date < day]
    votes = int(ensemble_votes(daily).iloc[-1])
    reg = trend_regime(votes)
    idx_date = str(daily.index[-1].date()) if len(daily) else ""
    lab = {1: "上涨趋势（核心持 SOXL / 禁空）", -1: "下跌趋势（空核心 / 只做空 SOXS）", 0: "震荡过渡（空核心 / 战术双向）"}[reg]
    return votes, reg, f"三票制 {idx_sym}({idx_date}) = {votes:+d} 票 → {lab}"


def tactical_step(settings: Settings, md: Market, broker, books: Books, st: TacState, now: datetime,
                  dry: bool = False) -> TacState:
    """在当前时间点做一次检查（守护进程每隔 poll_seconds 调一次）。"""
    t = swing_cfg(settings)["tactical"]
    ts_cfg = t.get("trend_switch") or {}
    dual = bool(ts_cfg.get("enabled"))
    day = date.fromisoformat(st.day)
    if st.done:
        return st
    window = market_window(day)
    if window is None or now >= window[2]:
        st.done = True
        return st
    session_start, regular_close, session_end = window
    if now < session_start:
        return st
    tactical_exit = min(_at(day, t["exit"]), regular_close - timedelta(minutes=5))

    def note(msg):
        stamp = now.astimezone(ET).strftime("%m-%d %H:%M")
        st.log.append(f"{stamp} {msg}")
        print(f"[战术层 {stamp} ET] {msg}")

    def in_ovn() -> bool:
        tt = now.astimezone(ET).time()
        return tt >= time(20, 0) or tt < time(4, 0)

    def affordable_shares(budget, px):
        offset = getattr(broker, "order_cfg", {}).get("limit_offset_bps", 0) / 1e4
        limit = round(px * (1 + offset), 2)
        return math.floor(budget / limit) if limit > 0 else 0

    def enter(sym, px, why, *, position_frac: float | None = None) -> bool:
        eq = system_equity(settings)
        target = position_frac if position_frac is not None else st.frac or t["position_frac"]
        f = min(target, t["max_frac_ovn"] if in_ovn() else t["max_frac"])
        qty = affordable_shares(eq * f, px)
        if qty < 1:
            note(f"本金 ${eq:,.0f} 太少，买不起 1 股 {sym}")
            return False
        if _submit(broker, sym, "BUY", qty, px, why):
            if not dry:
                books.add("tactical", sym, qty)
            st.symbol, st.qty, st.entry, st.stopped = sym, qty, px, False
            st.frac_total, st.last_add, st.adds, st.peak = qty * px / eq, px, 0, px
            st.trades += 1
            note(f"买入 {sym} {qty} 股 @ {px:.2f}，约 ${qty * px:,.0f}（{st.frac_total:.0%}，{why}）")
            return True
        return False

    def add(px):
        eq = system_equity(settings)
        f_add = min(t["add_frac"], t["max_frac"] - st.frac_total)
        q = math.floor(eq * f_add / px)
        if q < 1:
            return
        if _submit(broker, st.symbol, "BUY", q, px, f"盈利加仓（比上次买入价 {st.last_add:.2f} 涨 {px / st.last_add - 1:+.1%}）"):
            if not dry:
                books.add("tactical", st.symbol, q)
            st.entry = (st.entry * st.qty + px * q) / (st.qty + q)
            st.qty += q
            st.frac_total += q * px / eq
            st.last_add, st.adds = px, st.adds + 1
            note(f"加仓 {st.symbol} {q} 股 @ {px:.2f}，总仓位 {st.qty} 股 / {st.frac_total:.0%}，均价 {st.entry:.2f}")

    def exit_(px, why, *, stop_price=None):
        if _submit(broker, st.symbol, "SELL", st.qty, px, why, stop_price=stop_price):
            if not dry:
                books.add("tactical", st.symbol, -st.qty)
            pnl = (px - st.entry) * st.qty
            st.pnl += pnl
            _add_pnl(settings, pnl, st.day)
            note(f"卖出 {st.symbol} {st.qty} 股 @ {px:.2f}（{why}），本笔 {(px / st.entry - 1):+.1%} / ${pnl:+,.0f}")
            st.qty, st.frac_total, st.adds = 0, 0.0, 0

    def core_buy(cst: CoreState, core_sym: str, px: float, why: str) -> bool:
        eq = system_equity(settings)
        f = float(ts_cfg.get("core_frac", 1.0))
        qty = affordable_shares(eq * f, px)
        if qty < 1:
            note(f"[核心层] 本金 ${eq:,.0f} 太少，买不起 1 股 {core_sym}")
            return False
        if _submit(broker, core_sym, "BUY", qty, px, why):
            if not dry:
                books.add("core", core_sym, qty)
            cst.holding, cst.entry = True, px
            cst.peak, cst.peak_day, cst.peak_checked_at = 0.0, "", ""
            cst.stopped, cst.stopped_day = False, ""
            _save_core_state(settings.environment, cst)
            note(f"[核心层] 买入 {core_sym} {qty} 股 @ {px:.2f}，约 ${qty * px:,.0f}（{qty * px / eq:.0%}，{why}）")
            return True
        return False

    def core_sell(cst: CoreState, core_sym: str, qty: int, px: float, why: str, *, stop_price=None) -> bool:
        if _submit(broker, core_sym, "SELL", qty, px, why, stop_price=stop_price):
            if not dry:
                books.add("core", core_sym, -qty)
            basis = cst.entry or px
            pnl = (px - basis) * qty
            st.pnl += pnl
            _add_pnl(settings, pnl, st.day)
            cst.holding, cst.entry = False, 0.0
            cst.peak, cst.peak_day, cst.peak_checked_at = 0.0, "", ""
            if stop_price is not None:
                cst.stopped, cst.stopped_day = True, str(day)
            else:
                cst.stopped, cst.stopped_day = False, ""
            _save_core_state(settings.environment, cst)
            note(f"[核心层] 卖出 {core_sym} {qty} 股 @ {px:.2f}（{why}），本笔 {(px / basis - 1):+.1%} / ${pnl:+,.0f}")
            return True
        return False

    if not st.equity_start:
        st.equity_start = system_equity(settings)

    # 0) 三票制双层动态切换：先确定当天大趋势，并协调核心层与战术层
    cst = _load_core_state(settings.environment) if dual else None
    if dual:
        core_sym = ts_cfg.get("core_symbol", t["long"])
        core_start = _at(day, ts_cfg.get("core_session_start", "09:30"))
        if st.trend_regime is None:
            st.trend_votes, st.trend_regime, st.trend_note = eval_trend_switch(settings, md, day)
            cst.last_votes, cst.last_regime = st.trend_votes, st.trend_regime
            _save_core_state(settings.environment, cst)
            note(st.trend_note)
        core_qty = int(books.qty("core", core_sym))

        # 非上涨趋势：先清空遗留核心仓，再让战术层按方向门禁运行
        if st.trend_regime != 1:
            if core_qty > 0:
                quote = md.minute_quote(core_sym, None, now)
                if quote is None or quote[0] <= session_start or now - quote[0] > timedelta(minutes=2):
                    return st
                _, cpx, _ = quote
                if not core_sell(cst, core_sym, core_qty, cpx, f"趋势转弱（三票 {st.trend_votes:+d}），平掉核心仓"):
                    return st
            elif cst.stopped:
                cst.stopped, cst.stopped_day = False, ""
                _save_core_state(settings.environment, cst)
        else:
            # 上涨趋势：
            # A) 核心仓接管当天，使用前一晚夜盘开始后的日内最高价。
            if core_qty > 0 or not cst.stopped:
                st.core_active_today = True
                st.decided = True
                if core_qty > 0:
                    if cst.entry <= 0:
                        raise RuntimeError("核心持仓缺少入场均价，请核对本地账本与券商成交")
                    if cst.peak_day != st.day:
                        cst.peak, cst.peak_day, cst.peak_checked_at = 0.0, st.day, ""
                    since = datetime.fromisoformat(cst.peak_checked_at) if cst.peak_checked_at else session_start
                    quote = md.minute_quote(core_sym, since, now)
                    if quote is not None:
                        observed_at, cpx, high = quote
                        cst.peak = max(cst.peak, high)
                        stop_pct = float(ts_cfg["trail_pct"])
                        stop = cst.peak * (1 - stop_pct)
                        if now - observed_at <= timedelta(minutes=2) and cpx <= stop:
                            # Preserve the quote cursor on a failed submission so the same bar can be retried.
                            why = f"核心移动止损（日内最高 {cst.peak:.2f} 回撤 {stop_pct:.0%}，止损线 {stop:.2f}）"
                            st.done = core_sell(cst, core_sym, core_qty, cpx, why, stop_price=stop)
                            if not st.done:
                                _save_core_state(settings.environment, cst)
                            return st
                        cst.peak_checked_at = observed_at.isoformat()
                        _save_core_state(settings.environment, cst)
                    return st
                if now >= tactical_exit:
                    st.done = True
                elif now >= core_start:
                    if KILL_SWITCH.exists():
                        note("KILL_SWITCH 存在，核心层今天不开仓")
                        st.done = True
                        return st
                    _, cpx = md.last_price(core_sym)
                    core_buy(cst, core_sym, cpx, f"三票上涨趋势（{st.trend_votes:+d} 票）建核心仓")
                return st

            # B) 止损次日起，在配置时段检查核心买回。夜盘内仍允许顺势战术信号。
            if st.qty == 0 and st.trades == 0:
                if str(day) <= cst.stopped_day:
                    st.done = True
                    return st
                rth_rebuy = ts_cfg.get("rebuy_session") == "RTH"
                rebuy_start = core_start if rth_rebuy else session_start
                rebuy_end = tactical_exit if rth_rebuy else session_end
                if now >= rebuy_end:
                    st.done = True
                    return st
                if now >= rebuy_start:
                    if st.core_prev_close is None:
                        st.core_prev_close = md.prev_rth_close(core_sym, day)
                    since = max(rebuy_start, now.replace(second=0, microsecond=0) - timedelta(minutes=3))
                    quote = md.minute_quote(core_sym, since, now)
                    reb = float(ts_cfg.get("rebuy_pct", 0.05))
                else:
                    quote = None
                if quote is not None:
                    observed_at, cpx, _ = quote
                    if now - observed_at <= timedelta(minutes=2) and cpx >= st.core_prev_close * (1 + reb):
                        if KILL_SWITCH.exists():
                            note("KILL_SWITCH 存在，核心层不买回")
                            st.done = True
                            return st
                        if core_buy(cst, core_sym, cpx, f"较上一交易日常规盘收盘 {st.core_prev_close:.2f} 上涨 {reb:.0%} 买回"):
                            st.core_active_today = True
                            st.decided = True
                        return st
                if st.decided or now >= tactical_exit:
                    return st

    # 1) 还没进场：在各个检查时点判断
    if not st.decided:
        if KILL_SWITCH.exists():
            note("KILL_SWITCH 存在，今天不做")
            st.done = True
            return st
        pc = t.get("path") or {}
        lo_t, hi_t = _scan_window(t, day)
        if _continuous(t) and now > hi_t:
            wait_core_rebuy = dual and st.trend_regime == 1 and cst.stopped and str(day) > cst.stopped_day
            note("已过夜盘扫描窗口，盘前结束转盯核心反弹买回" if wait_core_rebuy else "已过夜盘扫描窗口，今天不做")
            st.decided = True
            if not wait_core_rebuy:
                st.done = True
            return st
        due = [c for c in t["checkpoints"] if now >= _at(day, c) and c not in st.evaluated]
        scanning = _continuous(t) and lo_t <= now <= hi_t
        if not due and not scanning:
            return st
        # 错过了多个时点（比如进程晚启动）只按最新的判断一次；连续扫描时非检查时点标成"扫描"
        cp = due[-1] if due else f"{now.astimezone(ET):%H:%M} 扫描"
        st.evaluated += due
        if now >= _at(day, t["last_reentry"]):
            note("已过入场时间窗口，今天不做")
            st.decided = st.done = True
            return st
        allowed, direction_note = tactical_directions(t, st.trend_regime if dual else None)
        macro_cfg = t.get("macro_short_gate") or {}
        if macro_cfg.get("enabled"):
            if st.macro_short_allowed is None:
                if hasattr(md, "macro_short_allowed"):
                    st.macro_short_allowed, st.macro_short_note = md.macro_short_allowed(day, macro_cfg)
                else:
                    st.macro_short_allowed, st.macro_short_note = eval_macro_short_gate(day, macro_cfg)
                if st.macro_short_note:
                    note(st.macro_short_note)
            if not st.macro_short_allowed:
                allowed = allowed - {"short"}
                if not allowed:
                    direction_note = "三票仅允许做空，但利率门控禁做空 SOXS"
                else:
                    direction_note = f"{direction_note}（利率门控禁做空 SOXS）"
        if st.gate != direction_note:
            st.gate = direction_note
            note(direction_note)
        prev = md.prev_rth_close(t["long"], day)
        _, px = md.last_price(t["long"])
        st.move = px / prev - 1
        why = f"{cp} 检查：{t['long']} 相对昨收 {st.move:+.1%}"
        is_rev = False
        if pc.get("enabled"):
            lo, hi, vwap = md.path_stats(t["long"], day)
            lo, hi = (px, px) if math.isnan(lo) else (min(lo, px), max(hi, px))
            side, desc, is_rev = path_signal(st.move, px, lo, hi, vwap, t["threshold"], pc)
            why += f"（{desc}）"
        else:
            side = 1 if st.move >= t["threshold"] else -1 if st.move <= -t["threshold"] else 0
        if side and t.get("korea_confirm") is not None and (not is_rev or pc.get("reversal_korea", True)):
            st.korea_move = md.kospi_move(now)
            why += f"，KOSPI {st.korea_move:+.1%}" if st.korea_move is not None else "，KOSPI 无数据"
            if st.korea_move is None or (st.korea_move > 0) != (side > 0) or abs(st.korea_move) < t["korea_confirm"]:
                side = 0
                why += " → 韩国不确认"
        if side and ("long" if side > 0 else "short") not in allowed:
            why += " → 方向被三票制或配置限制"
            side = 0
        last_cp = all(c in st.evaluated for c in t["checkpoints"]) and (not _continuous(t) or now >= hi_t)
        if not side:
            # 连续扫描每分钟都会走到这里：检查时点照常记一条，其余只在结论变化时记，避免刷屏
            tag = why.rsplit("→", 1)[-1] if "→" in why else "无信号"
            tag = "".join(ch for ch in tag if not (ch.isdigit() or ch in "+-.%）) "))   # 去掉会变的数字
            wait_core_rebuy = dual and st.trend_regime == 1 and cst.stopped and str(day) > cst.stopped_day
            if due or last_cp or tag != st.scan_note:
                tail = "，盘前结束转盯核心反弹买回" if (last_cp and wait_core_rebuy) else (
                    "，今天不做" if last_cp else "，继续等" if _continuous(t) else "，继续等下一个时点")
                note(why + tail)
            st.scan_note = tag
            if last_cp:
                st.decided = True
                if not wait_core_rebuy:
                    st.done = True
            return st
        st.decided, st.side = True, side
        er = None
        sz = t.get("sizing") or {}
        if sz.get("mode") == "dynamic" and sz.get("use_trend_efficiency", False):
            try:
                daily = md.daily(sz.get("trend_index", "SOXX"), 60)
                er = efficiency_ratio(daily.close, sz.get("er_window", 10))
            except Exception as e:
                note(f"取 {sz.get('trend_index', 'SOXX')} 日线失败，趋势效率按 1.0：{e}")
        st.frac, st.sizing_note = position_frac(t, st.move, st.korea_move, er)
        note(st.sizing_note)
        sym = t["long"] if side > 0 else t["short"]
        _, spx = md.last_price(sym)
        enter(sym, spx, why)
        if not st.qty:
            st.done = not (dual and st.trend_regime == 1 and cst.stopped and st.day > cst.stopped_day)
        return st

    sym = st.symbol or (t["long"] if st.side > 0 else t["short"])

    # 2) 收盘前平仓
    if now >= tactical_exit:
        if st.qty:
            _, px = md.last_price(sym)
            exit_(px, "收盘前平仓")
        st.done = True
        note(f"今天结束，战术层 ${st.pnl:+,.0f}，系统权益 ${system_equity(settings):,.0f}")
        return st

    # 3) 第二腿按本次买回后的最高价计算移动止损。
    reentry_trail = t.get("reentry_trail_pct")
    if st.qty and st.reentries and reentry_trail:
        since = datetime.fromisoformat(st.reentry_checked_at) if st.reentry_checked_at else None
        quote = md.minute_quote(sym, since, now)
        if quote is not None:
            observed_at, px, high = quote
            st.peak = max(st.peak, st.entry, high)
            stop = st.peak * (1 - reentry_trail)
            if now - observed_at <= timedelta(minutes=2) and px <= stop:
                exit_(px, f"第 2 腿移动止损：从买回后最高 {st.peak:.2f} 回撤 {reentry_trail:.0%}", stop_price=stop)
                if not st.qty:
                    st.stopped, st.low_after_stop = True, px
            else:
                st.reentry_checked_at = observed_at.isoformat()
        return st

    _, px = md.last_price(sym)
    if st.qty:
        st.peak = max(st.peak or px, px)
        rstop = t.get("reentry_stop_pct")
        if st.reentries >= 1 and rstop is not None:
            stop = st.entry * (1 - rstop)
            if rstop > 0 and px <= stop:
                exit_(px, f"第 2 腿兜底止损：跌破均价 {rstop:.0%}", stop_price=stop)
                st.stopped, st.low_after_stop = True, px
            return st
        if px <= st.entry * (1 - t["stop_pct"]):
            exit_(px, f"止损：跌破均价 {t['stop_pct']:.0%}", stop_price=st.entry * (1 - t["stop_pct"]))
            st.stopped, st.low_after_stop = True, px
        elif st.adds and t.get("trail_after_add") and px <= st.peak * (1 - t["trail_after_add"]):
            exit_(px, f"加仓后从最高 {st.peak:.2f} 回撤 {t['trail_after_add']:.0%} 止盈",
                  stop_price=st.peak * (1 - t["trail_after_add"]))
            st.stopped, st.low_after_stop = True, px
        elif t.get("trail_pct") and px <= st.peak * (1 - t["trail_pct"]):
            exit_(px, f"移动止损：从持仓最高 {st.peak:.2f} 回撤 {t['trail_pct']:.0%}",
                  stop_price=st.peak * (1 - t["trail_pct"]))
            st.stopped, st.low_after_stop = True, px
        elif (t.get("add_step") and not in_ovn() and _at(day, t["add_from"]) <= now < _at(day, t["add_last"])
              and st.frac_total < t["max_frac"] - 1e-6 and px >= st.last_add * (1 + t["add_step"])):
            add(px)
        return st

    # 4) 卖出后：从最低点反弹 rebound_pct 就按原方向买回（次数按 max_reentries，None = 不限）
    limit = t.get("max_reentries")
    if st.stopped and t["reentry"] and (limit is None or st.reentries < limit):
        st.low_after_stop = min(st.low_after_stop or px, px)
        if now < _at(day, t["last_reentry"]) and px >= st.low_after_stop * (1 + t["rebound_pct"]):
            why = f"卖出后从低点 {st.low_after_stop:.2f} 反弹 {t['rebound_pct']:.0%} 买回（今天第 {st.reentries + 1} 次）"
            if enter(sym, px, why, position_frac=t.get("reentry_position_frac")):
                st.reentries += 1
                # The entry minute can include a high reached before this buy.
                st.reentry_checked_at = (now.replace(second=0, microsecond=0) + timedelta(minutes=1)).isoformat()
    return st


def run_tactical_day(settings: Settings, md: Market, broker, books: Books, now_fn: Callable[[], datetime],
                     sleep_fn: Callable[[float], None] = _time.sleep, dry: bool = False,
                     day: date | None = None) -> TacState:
    """运行一个交易日。战术仓尾盘退出，核心仓继续监控到盘后关闭。"""
    t = swing_cfg(settings)["tactical"]
    ts_cfg = t.get("trend_switch") or {}
    day = day or trading_day(now_fn())
    st = _load_state(settings.environment, day)
    window = market_window(day)
    if window is None:
        st.done = True
        return st
    session_start, regular_close, session_end = window
    now = now_fn()
    if now < session_start or now >= session_end:
        return st
    if st.done and ts_cfg.get("enabled"):
        cst = _load_core_state(settings.environment)
        waiting_to_rebuy = st.trend_regime == 1 and st.trades == 0 and cst.stopped and st.day > cst.stopped_day
        if books.qty("core", ts_cfg.get("core_symbol", t["long"])) or waiting_to_rebuy:
            st.done = False
    if st.done:
        print(f"[战术层] {day} 已经跑完：{st.log[-1] if st.log else ''}")
        return st
    if ts_cfg.get("enabled"):
        print(f"[双层模式] 启用三票制动态切换（{ts_cfg.get('index', 'SOXX')}）："
              f"上涨持 {ts_cfg.get('core_symbol', 'SOXL')}（最高价回撤止损 {ts_cfg['trail_pct']:.0%} / 次日起较上一常规盘收盘价上涨 {ts_cfg.get('rebuy_pct', 0.05):.0%} 买回），"
              f"非上涨切战术层")
    if _continuous(t):
        lo_t, hi_t = _scan_window(t, day)
        print(f"[战术层] 连续扫描 {lo_t:%H:%M}–{hi_t:%H:%M} ET（每 {t['poll_seconds']} 秒），看昨收以来的高低点 / VWAP")
    flat = min(_at(day, t["exit"]), regular_close - timedelta(minutes=5))
    print(f"[战术层] 交易日 {day}：检查时点 {', '.join(t['checkpoints'])} ET，{flat:%H:%M} ET 平仓；"
          f"系统权益 ${system_equity(settings):,.0f}，每笔 {t['position_frac']:.0%}")
    while not st.done:
        now = now_fn()
        try:
            st = tactical_step(settings, md, broker, books, st, now, dry=dry)
        except Exception as e:                    # 网络 / 接口偶发错误：记下来，下一轮再试
            st.log.append(f"{now.astimezone(ET):%m-%d %H:%M} 出错，稍后重试：{type(e).__name__}: {e}")
            print(st.log[-1])
        _save_state(settings.environment, st)
        if st.done:
            break
        core_start = _at(day, ts_cfg.get("core_session_start", "09:30"))
        core_flat = (ts_cfg.get("enabled") and st.trend_regime == 1 and st.qty == 0
                     and st.trades == 0 and not books.qty("core", ts_cfg.get("core_symbol", t["long"])))
        waiting_to_rebuy = core_flat and _load_core_state(settings.environment).stopped
        if core_flat and st.decided and now < core_start and not waiting_to_rebuy:
            sleep_fn(max(1.0, (core_start - now_fn()).total_seconds()))
        elif not st.decided and not waiting_to_rebuy:
            pending = [_at(day, c) for c in t["checkpoints"] if c not in st.evaluated]
            nxt = min(pending) if pending else now + timedelta(seconds=t["poll_seconds"])
            if _continuous(t):                     # 扫描窗口内每 poll_seconds 看一次
                lo_t, hi_t = _scan_window(t, day)
                if now < hi_t:
                    nxt = min(nxt, max(lo_t, now + timedelta(seconds=t["poll_seconds"])))
            sleep_fn(max(1.0, (nxt - now_fn()).total_seconds()))
        else:
            sleep_fn(max(0.0, min(t["poll_seconds"], (session_end - now_fn()).total_seconds())))
    journal.record("swing_tactical", **{k: v for k, v in asdict(st).items() if k != "log"}, log=st.log)
    return st


def market_window(day: date) -> tuple[datetime, datetime, datetime] | None:
    """返回美东夜盘开盘、常规收盘、盘后收盘。"""
    from .scalp_data import session_bounds

    regular = session_bounds(day)
    if regular is None:
        return None
    regular_close = regular[1].to_pydatetime()
    return _at(day, "20:00"), regular_close, regular_close + timedelta(hours=4)


def next_market_window(now: datetime) -> tuple[date, datetime, datetime]:
    """当前或下一个交易日的连续交易窗口，跳过休市。"""
    day = now.astimezone(ET).date()
    while True:
        window = market_window(day)
        if window is not None and now < window[2]:
            return day, window[0], window[2]
        day += timedelta(days=1)


def run_tactical_watch(settings: Settings, md: Market, broker, books: Books,
                       now_fn: Callable[[], datetime], sleep_fn: Callable[[float], None] = _time.sleep) -> None:
    """在每个交易窗口运行策略，闭市时等待下一场夜盘。"""
    while True:
        now = now_fn()
        day, opens, closes = next_market_window(now)
        if now < opens:
            sleep_fn(opens.timestamp() - now.timestamp())
            continue
        run_tactical_day(settings, md, broker, books, now_fn, sleep_fn=sleep_fn, day=day)
        remaining = closes.timestamp() - now_fn().timestamp()
        if remaining > 0:
            sleep_fn(remaining)


def trading_day(now: datetime) -> date:
    return next_market_window(now)[0]
