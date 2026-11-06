"""Webull 官方 Python SDK 的薄封装。

系统里只有这个文件直接接触 SDK。UAT（沙盒）接入点的配置方式
参考了官方 webull-openapi-mcp 的 sdk_client.py。
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import pandas as pd

from .config import Settings

log = logging.getLogger(__name__)

# SDK 自带的 endpoints.json 只包含生产环境，沙盒需要手动注入
UAT_ENDPOINTS = {
    "us": {
        "api": "api.sandbox.webull.com",
        "quotes-api": "api.sandbox.webull.com",
        "events-api": "events-api.sandbox.webull.com",
    },
    "hk": {
        "api": "api.sandbox.webull.hk",
        "quotes-api": "api.sandbox.webull.hk",   # developer.webull.hk：行情 HTTP 与交易同一主机
        "events-api": "events-api.sandbox.webull.hk",
    },
}

# 系统交易美股，资金统一按美元计算（香港等多币种账户也一样）
BASE_CURRENCY = "USD"


def _json(resp: Any) -> Any:
    if resp is None:
        return None
    if hasattr(resp, "json") and callable(resp.json):
        return resp.json()
    return resp


def _unwrap_list(data: Any) -> list:
    """Webull 接口有时直接返回 list，有时包一层 {result|data|items: [...]}。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("result", "data", "items", "list"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


class WebullClient:
    def __init__(self, settings: Settings, interactive_auth: bool = False):
        if not settings.has_webull_credentials:
            raise RuntimeError("未配置 WEBULL_APP_KEY / WEBULL_APP_SECRET，请先复制 .env.example 为 .env 并填写")
        from webull.core.client import ApiClient
        from webull.core.common.api_type import DEFAULT, EVENTS, QUOTES
        from webull.data.data_client import DataClient
        from webull.trade.trade_client import TradeClient

        self.settings = settings
        api = ApiClient(
            settings.app_key,
            settings.app_secret,
            settings.region_id,
            token_check_duration_seconds=300 if interactive_auth else 15,
            token_check_interval_seconds=5 if interactive_auth else 3,
        )
        if settings.token_dir:
            from pathlib import Path
            Path(settings.token_dir).mkdir(parents=True, exist_ok=True)
            api.set_token_dir(settings.token_dir)
        if settings.environment == "uat":
            eps = UAT_ENDPOINTS.get(settings.region_id)
            if not eps:
                raise RuntimeError(f"暂未配置 {settings.region_id} 区域的沙盒地址")
            for key, api_type in (("api", DEFAULT), ("quotes-api", QUOTES), ("events-api", EVENTS)):
                api.add_endpoint(settings.region_id, eps[key], api_type)
        api.set_stream_logger(log_level=logging.CRITICAL, stream=sys.stderr)  # SDK 出错时会把整个请求打出来，太吵；错误由调用方处理
        self.trade = TradeClient(api)
        self.data = DataClient(api)
        self._account_id: str | None = None

    # ---------------- 行情 ----------------
    def daily_bars(self, symbols: list[str], count: int = 1200) -> dict[str, pd.DataFrame]:
        """批量拉取日线（前复权）。返回 {symbol: DataFrame[open,high,low,close,volume]}。"""
        out: dict[str, pd.DataFrame] = {}
        # 分批，每批最多 20 个
        for i in range(0, len(symbols), 20):
            batch = symbols[i : i + 20]
            data = _json(
                self.data.market_data.get_batch_history_bar(batch, "US_STOCK", "D", count=str(min(count, 1200)))
            )
            for item in _unwrap_list(data):
                sym = item.get("symbol")
                rows = item.get("result") or []
                if sym and rows:
                    out[sym] = bars_to_frame(rows)
        return out

    def snapshot(self, symbols: list[str]) -> dict[str, dict]:
        data = _json(self.data.market_data.get_snapshot(symbols, "US_STOCK"))
        return {d.get("symbol"): d for d in _unwrap_list(data) if d.get("symbol")}

    def stock_news(self, symbol: str, limit: int = 8) -> list[dict]:
        data = _json(self.data.news.get_stock_news(symbol, category="US_STOCK"))
        items = _unwrap_list(data)
        keep = ("title", "summary", "source_name", "source", "publish_time", "news_time", "sentiment")
        return [{k: it[k] for k in keep if k in it} for it in items[:limit]]

    # ---------------- 账户 ----------------
    @property
    def account_id(self) -> str:
        if self._account_id is None:
            accounts = _unwrap_list(_json(self.trade.account_v2.get_account_list()))
            if not accounts:
                raise RuntimeError("没有找到可用账户")
            self._account_id = str(accounts[0]["account_id"])
            log.info("使用账户 %s", self._account_id)
        return self._account_id

    def balance(self) -> dict:
        return _json(self.trade.account_v2.get_account_balance(self.account_id)) or {}

    def _currency_row(self, b: dict) -> dict | None:
        for row in b.get("account_currency_assets") or []:
            if row.get("currency") == BASE_CURRENCY:
                return row
        return None

    def cash(self) -> float:
        """可用美元。多币种账户读 USD 那一行的 buying_power；否则退回总额字段。"""
        b = self.balance()
        row = self._currency_row(b)
        if row is not None:
            return max(float(row.get("buying_power") or 0), 0.0)
        for key in ("buying_power", "total_cash_balance", "cash_balance"):
            if b.get(key) not in (None, ""):
                return float(b[key])
        return 0.0

    def equity(self) -> float:
        """美元权益。多币种账户 = USD 现金 + USD 市值；否则退回总额字段。"""
        b = self.balance()
        row = self._currency_row(b)
        if row is not None:
            return float(row.get("cash_balance") or 0) + float(row.get("market_value") or 0)
        for key in ("total_net_liquidation_value", "net_liquidation_value", "total_asset"):
            if b.get(key) not in (None, ""):
                return float(b[key])
        return float(b.get("total_cash_balance", 0)) + float(b.get("total_market_value", 0))

    def positions(self) -> dict[str, dict]:
        """{symbol: {quantity, cost_price, last_price}}，只保留股票。"""
        rows = _unwrap_list(_json(self.trade.account_v2.get_account_position(self.account_id)))
        out = {}
        for p in rows:
            if p.get("instrument_type", "EQUITY") not in ("EQUITY", "STOCK"):
                continue
            qty = float(p.get("quantity") or 0)
            if qty:
                out[p["symbol"]] = {
                    "quantity": qty,
                    "cost_price": float(p.get("cost_price") or 0),
                    "last_price": float(p.get("last_price") or 0),
                }
        return out

    def open_order_symbols(self) -> set[str]:
        data = _json(self.trade.order_v3.list_order_open(self.account_id))
        syms = set()
        for o in _unwrap_list(data):
            for leg in o.get("orders", [o]):
                if leg.get("symbol"):
                    syms.add(leg["symbol"])
        return syms

    def open_orders(self) -> list[dict]:
        return _unwrap_list(_json(self.trade.order_v3.list_order_open(self.account_id)))

    def order_history(self, days: int = 7) -> list[dict]:
        from datetime import date, timedelta
        start = str(date.today() - timedelta(days=days))
        return _unwrap_list(_json(self.trade.order_v3.get_order_history(self.account_id, page_size=100, start_date=start)))

    # ---------------- 下单 ----------------
    def preview_order(self, order: dict) -> Any:
        return _json(self.trade.order_v3.preview_order(account_id=self.account_id, preview_orders=[order]))

    def place_order(self, order: dict) -> Any:
        return _json(self.trade.order_v3.place_order(account_id=self.account_id, new_orders=[order]))


def bars_to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    t = df["time"]
    if pd.api.types.is_numeric_dtype(t) or t.astype(str).str.fullmatch(r"\d+").all():
        idx = pd.to_datetime(t.astype("int64"), unit="ms", utc=True)
    else:
        idx = pd.to_datetime(t, utc=True)
    df.index = idx.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()
