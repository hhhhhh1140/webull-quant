"""Read-only OpenD checks, live Futu market data, and Futu order brokers for the swing strategy."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from hashlib import sha256
import math
import threading
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from .broker import build_webull_order
from .risk import OrderIntent

ET = ZoneInfo("America/New_York")


def _futu():
    try:
        import futu
    except ImportError as exc:
        raise RuntimeError("富途接入需要安装 futu-api") from exc
    return futu


def _trade_context():
    futu = _futu()
    return futu.OpenSecTradeContext(
        filter_trdmarket=futu.TrdMarket.US,
        host="127.0.0.1",
        port=11111,
        security_firm=futu.SecurityFirm.FUTUSECURITIES,
    )


def us_accounts(context):
    ret, rows = context.get_acc_list()
    if ret != _futu().RET_OK:
        raise RuntimeError(f"富途 OpenD 无法读取账户列表：{rows}")
    return [r for r in rows.to_dict("records") if r.get("acc_status") == "ACTIVE"
            and "US" in r.get("trdmarket_auth", [])]


def select_account(accounts, environment: str, account_id: int | None = None):
    matches = [r for r in accounts if r["trd_env"] == environment
               and (account_id is None or int(r["acc_id"]) == account_id)]
    if len(matches) != 1:
        raise RuntimeError(f"富途 {environment} 需要唯一美股账户，当前匹配 {len(matches)} 个")
    return matches[0]


def check_accounts() -> int:
    context = _trade_context()
    try:
        accounts = us_accounts(context)
    finally:
        context.close()
    for account in accounts:
        print(f"富途美股 {account['trd_env']} 账户尾号 {str(account['acc_id'])[-4:]} "
              f"{account.get('acc_type', '')}")
    return 0 if accounts else 1


class FutuShadowBroker:
    """Record the same strategy using Futu's order fields and its own simulated ledger."""

    name = "futu-shadow"

    def __init__(self, order_cfg, *, environment="REAL", account_id=None):
        self.order_cfg = order_cfg
        context = _trade_context()
        try:
            self.account = select_account(us_accounts(context), environment, account_id)
        finally:
            context.close()
        fingerprint = sha256(str(self.account["acc_id"]).encode()).hexdigest()[:10]
        self.book_environment = f"futu_{environment.lower()}_shadow_{fingerprint}"

    def submit(self, intent: OrderIntent) -> dict:
        # Reuse the existing order pricing and round to the same tick as Webull.
        reference = build_webull_order(intent, {**self.order_cfg, "type": "LIMIT"})
        session = {"CORE": "RTH", "ALL": "ETH", "NIGHT": "OVERNIGHT", "ALL_DAY": "ALL"}[
            reference["support_trading_session"]
        ]
        order = {
            "symbol": f"US.{intent.symbol}",
            "side": intent.side,
            "quantity": int(intent.quantity),
            "limit_price": reference["limit_price"],
            "order_type": "NORMAL",
            "session": session,
            "time_in_force": reference["time_in_force"],
            "trd_env": self.account["trd_env"],
            "account_suffix": str(self.account["acc_id"])[-4:],
        }
        print(f"  [FUTU SHADOW] {order}")
        return {"status": "dry_run", "order": order}


class FutuSimBroker:
    """Submit strategy orders to the Futu US SIMULATE account (never REAL).

    If Futu's paper trading endpoint rejects an extended/overnight session order and
    ``fallback_local`` is True, records a simulated local fill so the strategy's
    state machine and ledger stay in sync without missing overnight signals.
    """

    name = "futu-sim"

    def __init__(self, order_cfg, *, account_id=None, label: str = "futu-sim", fallback_local: bool = True):
        self.order_cfg = order_cfg
        self.label = label
        self.fallback_local = fallback_local
        context = _trade_context()
        try:
            self.account = select_account(us_accounts(context), "SIMULATE", account_id)
        finally:
            context.close()
        fingerprint = sha256(str(self.account["acc_id"]).encode()).hexdigest()[:10]
        self.book_environment = f"futu_sim_{fingerprint}"

    def submit(self, intent: OrderIntent) -> dict:
        futu = _futu()
        reference = build_webull_order(intent, {**self.order_cfg, "type": "LIMIT"})
        session_name = {"CORE": "RTH", "ALL": "ETH", "NIGHT": "OVERNIGHT", "ALL_DAY": "ALL"}[
            reference["support_trading_session"]
        ]
        session_enum = getattr(futu.Session, session_name)
        trd_side = futu.TrdSide.BUY if intent.side.upper() == "BUY" else futu.TrdSide.SELL
        limit_price = float(reference["limit_price"])
        code = f"US.{intent.symbol.upper()}"
        order_meta = {
            "symbol": code,
            "side": intent.side,
            "quantity": int(intent.quantity),
            "limit_price": reference["limit_price"],
            "order_type": "NORMAL",
            "session": session_name,
            "time_in_force": reference["time_in_force"],
            "trd_env": "SIMULATE",
            "account_suffix": str(self.account["acc_id"])[-4:],
        }
        context = _trade_context()
        try:
            ret, data = context.place_order(
                price=limit_price,
                qty=int(intent.quantity),
                code=code,
                trd_side=trd_side,
                order_type=futu.OrderType.NORMAL,
                trd_env=futu.TrdEnv.SIMULATE,
                acc_id=int(self.account["acc_id"]),
                time_in_force=futu.TimeInForce.DAY,
                session=session_enum,
                remark=f"wquant:{self.label}"[:60],
            )
        finally:
            context.close()
        if ret == futu.RET_OK:
            order_id = str(data["order_id"].iloc[0]) if hasattr(data, "iloc") and len(data) else ""
            order_meta["order_id"] = order_id
            print(f"  [FUTU SIM:{self.label}] 已报模拟单 {order_meta}")
            return {"status": "submitted", "order": order_meta}
        msg = str(data)
        if self.fallback_local:
            order_meta["futu_sim_rejected"] = msg
            print(f"  [FUTU SIM:{self.label}] 富途模拟盘拒单（{msg}），改用本地模拟记账 @ {limit_price:.2f}")
            return {"status": "filled", "order": order_meta, "fallback": "local"}
        print(f"  [FUTU SIM:{self.label}] 富途模拟盘拒单：{msg}")
        return {"status": "rejected", "order": order_meta, "reason": msg}


def _classify_et_session(idx_et: pd.DatetimeIndex) -> list[str]:
    mins = idx_et.hour * 60 + idx_et.minute
    return [
        "OVN" if (m >= 20 * 60 or m < 4 * 60)
        else "PRE" if m < 9 * 60 + 30
        else "RTH" if m < 16 * 60
        else "ATH"
        for m in mins
    ]


def _futu_kline_to_frame(df: pd.DataFrame, step_minutes: int) -> pd.DataFrame:
    """Convert Futu K-line DataFrame (NY bar-end time_key) to bar-start ET OHLCV + session."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "session"])
    naive = pd.to_datetime(df["time_key"])
    step = pd.Timedelta(minutes=step_minutes)
    utc_starts = []
    prev_utc = None
    for t in naive:
        try:
            loc = t.tz_localize(ET, ambiguous="raise", nonexistent="shift_forward")
        except Exception:
            c1 = t.tz_localize(ET, ambiguous=True).tz_convert("UTC")
            c2 = t.tz_localize(ET, ambiguous=False).tz_convert("UTC")
            loc = (c2 if prev_utc is not None and c1 <= prev_utc else c1).tz_convert(ET)
        u = loc.tz_convert("UTC")
        utc_starts.append(u - step)
        prev_utc = u
    idx_et = pd.DatetimeIndex(utc_starts, name="t").tz_convert(ET)
    out = pd.DataFrame(
        {
            "open": df["open"].astype(float).values,
            "high": df["high"].astype(float).values,
            "low": df["low"].astype(float).values,
            "close": df["close"].astype(float).values,
            "volume": df["volume"].astype(float).values,
            "session": _classify_et_session(idx_et),
        },
        index=idx_et,
    )
    return out.loc[~out.index.duplicated(keep="last")].sort_index()


def sanitize_futu_0400_spikes(df: pd.DataFrame, prev_rth_closes: dict[date, float] | None = None) -> pd.DataFrame:
    """Remove Futu's 04:00 ET pre-market open spike caused by prior-day RTH close reporting.

    When the 04:00 bar's open matches the prior day's RTH close (within 0.5%) while jumping
    more than 1.5% away from the immediately preceding OVN bar's close, realign open/high/low
    so overnight trailing stops and path filters are not falsely triggered.
    """
    if df.empty:
        return df
    is_0400 = (df.index.hour == 4) & (df.index.minute <= 1)
    if not is_0400.any():
        return df
    out = df.copy()
    rth_by_day: dict[date, float] = dict(prev_rth_closes or {})
    rth = out[out["session"] == "RTH"]
    if not rth.empty:
        for d, grp in rth.groupby(rth.index.date):
            rth_by_day.setdefault(d, float(grp["close"].iloc[-1]))
    sorted_rth_days = sorted(rth_by_day)

    positions = [i for i, flag in enumerate(is_0400) if flag]
    for pos in positions:
        if pos == 0:
            continue
        ts = out.index[pos]
        prev_bar = out.iloc[pos - 1]
        if prev_bar["session"] != "OVN" or (ts - out.index[pos - 1]) > pd.Timedelta(minutes=15):
            continue
        prior_days = [d for d in sorted_rth_days if d < ts.date()]
        if not prior_days:
            continue
        prth = rth_by_day[prior_days[-1]]
        c_ovn = float(prev_bar["close"])
        row = out.iloc[pos]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        if prth > 0 and c_ovn > 0 and abs(o / prth - 1.0) < 0.005 and abs(o / c_ovn - 1.0) > 0.015:
            new_o = c_ovn
            new_l = min(new_o, c) if abs(l / o - 1.0) < 0.002 else l
            new_h = max(new_o, c) if abs(h / o - 1.0) < 0.002 else h
            out.iat[pos, out.columns.get_loc("open")] = new_o
            out.iat[pos, out.columns.get_loc("high")] = max(new_h, new_o, c)
            out.iat[pos, out.columns.get_loc("low")] = min(new_l, new_o, c)
    return out


class FutuMarket:
    """Live Market adapter backed by local Futu OpenD (24h US 1m/5m/daily + Yahoo KOSPI/macro)."""

    def __init__(self, now: Callable[[], datetime], *, host: str = "127.0.0.1", port: int = 11111):
        self.now = now
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._ctx = None
        self._subscribed: set[tuple[str, str]] = set()
        self._prev_rth_cache: dict[tuple[str, date], float] = {}

    def _quote_ctx(self):
        if self._ctx is None:
            futu = _futu()
            self._ctx = futu.OpenQuoteContext(host=self.host, port=self.port)
            self._subscribed.clear()
        return self._ctx

    def close(self) -> None:
        with self._lock:
            if self._ctx is not None:
                try:
                    self._ctx.close()
                finally:
                    self._ctx = None
                    self._subscribed.clear()

    def _reset_ctx(self) -> None:
        if self._ctx is not None:
            try:
                self._ctx.close()
            except Exception:
                pass
            self._ctx = None
            self._subscribed.clear()

    def _ensure_subscribed(self, code: str, subtype_name: str) -> None:
        key = (code, subtype_name)
        if key in self._subscribed:
            return
        futu = _futu()
        ctx = self._quote_ctx()
        subtype = getattr(futu.SubType, subtype_name)
        ret, err = ctx.subscribe([code], [subtype], subscribe_push=False, session=futu.Session.ALL)
        if ret != futu.RET_OK:
            raise RuntimeError(f"富途订阅 {code} {subtype_name} 失败：{err}")
        self._subscribed.add(key)

    def _cur_bars(self, sym: str, step_minutes: int, count: int) -> pd.DataFrame:
        futu = _futu()
        code = f"US.{sym.upper()}"
        sub_name = "K_1M" if step_minutes == 1 else "K_5M"
        ktype = futu.SubType.K_1M if step_minutes == 1 else futu.SubType.K_5M
        with self._lock:
            for attempt in range(2):
                try:
                    self._ensure_subscribed(code, sub_name)
                    ret, data = self._quote_ctx().get_cur_kline(code, min(count, 1000), ktype, futu.AuType.NONE)
                    if ret != futu.RET_OK:
                        raise RuntimeError(f"富途读取 {code} {sub_name} 失败：{data}")
                    return _futu_kline_to_frame(data, step_minutes)
                except Exception:
                    self._reset_ctx()
                    if attempt == 1:
                        raise
        return pd.DataFrame()

    def _history_bars(self, sym: str, step_minutes: int, start_day: date, end_day: date) -> pd.DataFrame:
        futu = _futu()
        code = f"US.{sym.upper()}"
        ktype = futu.KLType.K_1M if step_minutes == 1 else futu.KLType.K_5M
        parts = []
        page_key = None
        with self._lock:
            for attempt in range(2):
                try:
                    ctx = self._quote_ctx()
                    while True:
                        ret, data, page_key = ctx.request_history_kline(
                            code,
                            start=start_day.isoformat(),
                            end=end_day.isoformat(),
                            ktype=ktype,
                            autype=futu.AuType.NONE,
                            max_count=1000,
                            page_req_key=page_key,
                            session=futu.Session.ALL,
                        )
                        if ret != futu.RET_OK:
                            raise RuntimeError(f"富途历史 K 线 {code} 失败：{data}")
                        if len(data):
                            parts.append(data)
                        if page_key is None or len(parts) >= 4:
                            break
                    break
                except Exception:
                    self._reset_ctx()
                    parts.clear()
                    page_key = None
                    if attempt == 1:
                        raise
        if not parts:
            return pd.DataFrame()
        return _futu_kline_to_frame(pd.concat(parts, ignore_index=True), step_minutes)

    def last_price(self, sym: str) -> tuple[datetime, float]:
        b = self._cur_bars(sym, 1, 5)
        if b.empty:
            raise RuntimeError(f"富途未返回 {sym} 的 1 分钟行情")
        return b.index[-1].to_pydatetime(), float(b["close"].iloc[-1])

    def _prev_rth_map(self, sym: str, ref_day: date) -> dict[date, float]:
        try:
            px = self.prev_rth_close(sym, ref_day)
            return {ref_day - timedelta(days=1): px}
        except Exception:
            return {}

    def minute_bars(self, sym: str, since: datetime, now: datetime) -> pd.DataFrame:
        """返回富途已收盘 1 分钟 OHLCV（bar-start 美东索引，已过滤 04:00/04:01 昨收假针）。"""
        minute = timedelta(minutes=1)
        need = max(30, math.ceil((now - since) / minute) + 10)
        bars = self._cur_bars(sym, 1, min(need, 1000))
        if bars.empty or bars.index[0] > since:
            hist = self._history_bars(sym, 1, since.astimezone(ET).date() - timedelta(days=1), now.astimezone(ET).date())
            if not hist.empty:
                bars = pd.concat([hist, bars])
                bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index()
        if bars.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "session"])
        bars = sanitize_futu_0400_spikes(bars, self._prev_rth_map(sym, now.astimezone(ET).date()))
        return bars.loc[(bars.index >= since) & (bars.index + minute <= now)]

    def minute_quote(self, sym: str, since: datetime | None, now: datetime) -> tuple[datetime, float, float] | None:
        """返回富途已收盘 1 分钟线的最后收盘时刻、收盘价及区间最高价（已过滤 04:00 昨收假针）。"""
        minute = timedelta(minutes=1)
        need = 5 if since is None else max(5, math.ceil((now - since) / minute) + 5)
        bars = self._cur_bars(sym, 1, min(need, 1000))
        if since is not None and (bars.empty or bars.index[0] > since):
            hist = self._history_bars(sym, 1, since.astimezone(ET).date() - timedelta(days=1), now.astimezone(ET).date())
            if not hist.empty:
                bars = pd.concat([hist, bars])
                bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index()
        if bars.empty:
            return None
        bars = sanitize_futu_0400_spikes(bars, self._prev_rth_map(sym, now.astimezone(ET).date()))
        bars = bars.loc[bars.index + minute <= now]
        bars = bars.iloc[-1:] if since is None else bars.loc[bars.index >= since]
        if bars.empty:
            return None
        return (bars.index[-1].to_pydatetime() + minute, float(bars["close"].iloc[-1]), float(bars["high"].max()))

    def prev_rth_close(self, sym: str, day: date) -> float:
        key = (sym.upper(), day)
        if key in self._prev_rth_cache:
            return self._prev_rth_cache[key]
        b = self._cur_bars(sym, 5, 600)
        rth = b[(b["session"] == "RTH") & (b.index.date < day)] if not b.empty else pd.DataFrame()
        if rth.empty:
            b = self._history_bars(sym, 5, day - timedelta(days=7), day)
            rth = b[(b["session"] == "RTH") & (b.index.date < day)] if not b.empty else pd.DataFrame()
        if rth.empty:
            raise RuntimeError(f"富途无法找到 {sym} 在 {day} 之前的常规盘收盘价")
        val = float(rth["close"].iloc[-1])
        self._prev_rth_cache[key] = val
        return val

    def path_stats(self, sym: str, day: date) -> tuple[float, float, float | None]:
        """昨天常规时段收盘以来（盘后 + 夜盘 + 盘前 + 今天）的最低、最高和 VWAP，用富途 5 分钟线。"""
        b = self._cur_bars(sym, 5, 600)
        rth = b[(b["session"] == "RTH") & (b.index.date < day)] if not b.empty else pd.DataFrame()
        if rth.empty:
            b = self._history_bars(sym, 5, day - timedelta(days=7), day)
            rth = b[(b["session"] == "RTH") & (b.index.date < day)] if not b.empty else pd.DataFrame()
        if rth.empty:
            return float("nan"), float("nan"), None
        b = sanitize_futu_0400_spikes(b)
        w = b[b.index > rth.index[-1]]
        if w.empty:
            return float("nan"), float("nan"), None
        vol = w["volume"].sum()
        vwap = float(((w["high"] + w["low"] + w["close"]) / 3 * w["volume"]).sum() / vol) if vol > 0 else None
        return float(w["low"].min()), float(w["high"].max()), vwap

    def daily(self, sym: str, count: int) -> pd.DataFrame:
        futu = _futu()
        code = f"US.{sym.upper()}"
        now = self.now().astimezone(ET)
        end_d = now.date()
        start_d = end_d - timedelta(days=max(int(count * 1.8), 60))
        with self._lock:
            for attempt in range(2):
                try:
                    ret, data, _ = self._quote_ctx().request_history_kline(
                        code,
                        start=start_d.isoformat(),
                        end=end_d.isoformat(),
                        ktype=futu.KLType.K_DAY,
                        autype=futu.AuType.QFQ,
                        max_count=max(count + 20, 200),
                    )
                    if ret != futu.RET_OK:
                        raise RuntimeError(f"富途日线 {code} 失败：{data}")
                    break
                except Exception:
                    self._reset_ctx()
                    if attempt == 1:
                        raise
        if data is None or data.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        idx = pd.DatetimeIndex(pd.to_datetime(data["time_key"]).dt.date).tz_localize(ET)
        df = pd.DataFrame(
            {
                "open": data["open"].astype(float).values,
                "high": data["high"].astype(float).values,
                "low": data["low"].astype(float).values,
                "close": data["close"].astype(float).values,
                "volume": data["volume"].astype(float).values,
            },
            index=idx,
        )
        df = df.loc[~df.index.duplicated(keep="last")].sort_index()
        if now.time() < time(16, 5):
            df = df[df.index.date < now.date()]
        return df.tail(count)

    def kospi_move(self, now: datetime) -> float | None:
        from .swing_live import fetch_kospi_move
        return fetch_kospi_move(now)

    def macro_short_allowed(self, day: date, gate_cfg: dict) -> tuple[bool, str]:
        from .swing_live import eval_macro_short_gate
        return eval_macro_short_gate(day, gate_cfg)
