"""本地网页仪表盘：python -m wquant dashboard

只监听 127.0.0.1，只读：查询账户、订单、日志和决策文件，不会下单，也不改任何配置。
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from . import journal
from .config import ROOT, Settings

HTML = Path(__file__).with_name("dashboard.html")
SNAPSHOT_EVERY = 600   # 权益快照间隔（秒）


# ---------------------------------------------------------------- 成交与盈亏
def _fills_path(env: str) -> Path:
    return journal.STATE / f"fills_{env}.json"


def _merge_fills(env: str, orders: list[dict]) -> list[dict]:
    """把接口返回的已成交订单合并进本地缓存（接口只能查近几天）。"""
    p = _fills_path(env)
    cache = {f["order_id"]: f for f in json.loads(p.read_text())} if p.exists() else {}
    for o in orders:
        q = float(o.get("filled_quantity") or 0)
        if q > 0 and o.get("order_id"):
            cache[o["order_id"]] = {
                "order_id": o["order_id"], "symbol": o["symbol"], "side": o["side"], "qty": q,
                "price": float(o.get("filled_price") or 0), "time": o.get("place_time_at", ""),
            }
    fills = sorted(cache.values(), key=lambda f: f["time"])
    journal.STATE.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(fills, indent=1))
    return fills


def realized_pnl(fills: list[dict]) -> tuple[float, list[dict]]:
    """先进先出计算已实现盈亏，并返回每笔卖出的盈亏。"""
    lots: dict[str, list[list[float]]] = {}
    total, closed = 0.0, []
    for f in fills:
        book = lots.setdefault(f["symbol"], [])
        if f["side"] == "BUY":
            book.append([f["qty"], f["price"]])
            continue
        remaining, pnl = f["qty"], 0.0
        while remaining > 1e-9 and book:
            lot = book[0]
            take = min(lot[0], remaining)
            pnl += take * (f["price"] - lot[1])
            lot[0] -= take
            remaining -= take
            if lot[0] <= 1e-9:
                book.pop(0)
        total += pnl
        closed.append({**f, "pnl": round(pnl, 2)})
    return round(total, 2), closed


# ---------------------------------------------------------------- 各部分数据
def _decisions_history(limit_days: int = 30) -> list[dict]:
    from .engine import DECISIONS
    outcomes: dict[tuple[str, str], dict] = {}
    for f in sorted(journal.LOGS.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("event") in ("decision", "order") and r.get("symbol"):
                outcomes[(f.stem, r["symbol"])] = {"outcome": r.get("outcome"), "order": r.get("order"),
                                                   "risk_notes": r.get("risk_notes")}
    days = []
    for c in sorted(DECISIONS.glob("*/candidates.json"))[-limit_days:]:
        day = c.parent.name
        packet = json.loads(c.read_text(encoding="utf-8"))
        dpath = c.parent / "decisions.json"
        decs = {}
        reviewer, view, hedges = None, None, []
        if dpath.exists():
            d = json.loads(dpath.read_text(encoding="utf-8"))
            reviewer = d.get("reviewer")
            view = d.get("market_view")
            hedges = [{**h, **outcomes.get((day, str(h.get("symbol", "")).upper()), {})} for h in d.get("hedges", [])]
            decs = {x.get("symbol", "").upper(): x for x in d.get("decisions", [])}
        items = []
        for cand in packet["candidates"]:
            s = cand["symbol"]
            items.append({
                "symbol": s, "action": cand["action"], "price": cand["price"], "rule_reason": cand["rule_reason"],
                "review": decs.get(s), **outcomes.get((day, s), {}),
            })
        days.append({"date": day, "environment": packet.get("environment"), "reviewer": reviewer, "items": items,
                     "market_view": view, "hedges": hedges, "sentiment": packet.get("sentiment"),
                     "blocked_buys": packet.get("blocked_buys", [])})
    return list(reversed(days))


_bt_cache: dict = {}


def _backtest(settings: Settings) -> dict | None:
    """用本地缓存的日线跑一次规则层回测，结果按缓存文件时间缓存。"""
    from .backtest import buy_and_hold, run_backtest
    from .data import CACHE, load_bars
    files = sorted(CACHE.glob("*.csv"))
    if not files:
        return None
    key = max(f.stat().st_mtime for f in files)
    if _bt_cache.get("key") == key:
        return _bt_cache["value"]
    strat = settings.section("strategy")
    index_syms = [x.upper() for x in (strat.get("market_filter") or {}).get("symbols", [])]
    bars = load_bars(settings, list(dict.fromkeys(settings.universe + index_syms)), source="cache")
    if not bars:
        return None
    bt = settings.section("backtest")
    res = run_backtest(bars, strat, settings.section("risk"), bt, universe=settings.universe, groups=settings.groups,
                       instruments={x: settings.instrument(x) for x in settings.universe})
    bench = buy_and_hold({k: v for k, v in bars.items() if k in settings.universe}, bt.get("initial_cash", 100_000))
    step = max(len(res.equity) // 300, 1)   # 抽样到约 300 个点
    eq = res.equity.iloc[::step]
    bm = bench.reindex(eq.index, method="ffill")
    value = {
        "metrics": res.metrics,
        "bench_return": round(float(bench.iloc[-1] / bench.iloc[0] - 1), 4),
        "bench_max_dd": round(float((bench / bench.cummax() - 1).min()), 4),
        "series": [{"date": str(d.date()), "strategy": round(float(v), 2),
                    "benchmark": None if pd.isna(b) else round(float(b), 2)}
                   for d, v, b in zip(eq.index, eq.values, bm.values)],
    }
    _bt_cache.update(key=key, value=value)
    return value


def _stop_levels(settings: Settings, symbols: list[str]) -> dict[str, dict]:
    from .data import CACHE, load_csv
    from .indicators import add_indicators
    strat = settings.section("strategy")
    highs = journal.trail_highs()
    out = {}
    for s in symbols:
        p = CACHE / f"{s}.csv"
        if not p.exists():
            continue
        df = add_indicators(load_csv(p), strat)
        row = df.iloc[-1]
        trail = max(highs.get(s, 0), float(row["close"]))
        inv = settings.instrument(s)["inverse"]
        ir = strat.get("inverse_rule") or {}
        mult = ir.get("atr_stop_mult", 2.0) if inv else strat.get("atr_stop_mult", 2.5)
        out[s] = {
            "atr_stop": round(trail - mult * float(row["atr"]), 2),
            "sma_fast": round(float(row["st_fast"] if inv else row["sma_fast"]), 2),
            "trail_high": round(trail, 2),
            "bar_date": str(df.index[-1].date()),
        }
    return out


def _account(settings: Settings) -> dict:
    env = settings.environment
    cap = float(settings.section("risk").get("max_capital") or settings.section("backtest").get("initial_cash", 100_000))
    if settings.has_webull_credentials:
        from .cli import _my_order_ids
        from .engine import _own_positions
        from .webull_client import WebullClient
        c = WebullClient(settings)
        busy = c.open_order_symbols()
        raw = c.positions()
        own = _own_positions(env, raw, busy) if settings.section("risk").get("manage_only_own_positions", True) else raw
        mine = _my_order_ids()
        orders, seen = [], set()
        for grp in c.open_orders() + c.order_history(30):
            for o in grp.get("orders", [grp]):
                if o.get("client_order_id") in mine and o.get("order_id") not in seen:
                    seen.add(o.get("order_id"))
                    orders.append({k: o.get(k) for k in ("symbol", "side", "status", "total_quantity", "filled_quantity",
                                                         "filled_price", "limit_price", "support_trading_session",
                                                         "place_time_at", "order_id")})
        fills = _merge_fills(env, orders)
        source = f"Webull {env.upper()} · 账户 {c.account_id}"
    else:
        from .broker import PaperBroker
        paper = PaperBroker({})
        own = {s: {**p, "last_price": p.get("cost_price", 0)} for s, p in paper.positions().items()}
        orders, fills, source = [], [], "本地模拟账户（未配置 Webull 密钥）"

    positions = []
    for s, p in own.items():
        last = float(p.get("last_price") or p["cost_price"])
        cost = float(p["cost_price"])
        qty = float(p["quantity"])
        positions.append({"symbol": s, "quantity": qty, "cost_price": cost, "last_price": last,
                          "market_value": round(qty * last, 2), "cost_value": round(qty * cost, 2),
                          "pnl": round(qty * (last - cost), 2), "pnl_pct": round(last / cost - 1, 4) if cost else 0})
    stops = _stop_levels(settings, [p["symbol"] for p in positions])
    # 仪表盘有 Webull 密钥时读的是 Webull 账户，对冲记录按环境存；否则读本地模拟账户
    open_hedges = journal.hedges(settings.environment if settings.has_webull_credentials else "paper")
    for p in positions:
        p["stop"] = stops.get(p["symbol"])
        p["hedge"] = open_hedges.get(p["symbol"])
        p["group"] = settings.groups.get(p["symbol"])

    realized, closed = realized_pnl(fills)
    invested = sum(p["cost_value"] for p in positions)
    mv = sum(p["market_value"] for p in positions)
    unrealized = round(mv - invested, 2)
    equity = round(cap + realized + unrealized, 2)
    snap = {"ts": datetime.now().isoformat(timespec="seconds"), "equity": equity, "market_value": round(mv, 2),
            "realized": realized, "unrealized": unrealized}
    history = _snapshot(env, snap)
    return {"source": source, "capital": cap, "equity": equity, "cash": round(cap + realized - invested, 2),
            "market_value": round(mv, 2), "invested": round(invested, 2), "unrealized": unrealized,
            "realized": realized, "positions": sorted(positions, key=lambda x: -x["market_value"]),
            "orders": sorted(orders, key=lambda o: o.get("place_time_at") or "", reverse=True),
            "closed_trades": closed, "equity_history": history}


def _snapshot(env: str, snap: dict) -> list[dict]:
    p = journal.STATE / f"equity_{env}.jsonl"
    hist = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    last_ts = datetime.fromisoformat(hist[-1]["ts"]).timestamp() if hist else 0
    if time.time() - last_ts >= SNAPSHOT_EVERY:
        journal.STATE.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(snap) + "\n")
        return (hist + [snap])[-2000:]
    return hist[-2000:] + [snap]   # 最新点只用于显示，不落盘


def _market(settings: Settings) -> dict | None:
    """用本地缓存日线计算大盘状态（scan / fetch 时会更新缓存）。"""
    from .data import CACHE, load_csv
    from .indicators import add_indicators
    from .strategy import market_regime
    strat = settings.section("strategy")
    syms = (strat.get("market_filter") or {}).get("symbols", [])
    frames = {s: add_indicators(load_csv(CACHE / f"{s}.csv"), strat) for s in syms if (CACHE / f"{s}.csv").exists()}
    out = market_regime(frames, strat)
    if frames:
        out["bar_date"] = str(max(f.index[-1] for f in frames.values()).date())
    return out


def collect(settings: Settings) -> dict:
    data = {"generated_at": datetime.now().isoformat(timespec="seconds"), "environment": settings.environment,
            "market": _market(settings),
            "region": settings.region_id, "mode": settings.mode, "ai_mode": settings.section("ai").get("mode"),
            "live_trading_enabled": settings.live_trading_enabled, "universe": settings.universe,
            "risk": settings.section("risk"), "kill_switch": (ROOT / "KILL_SWITCH").exists()}
    for key, fn in (("account", lambda: _account(settings)), ("decisions", _decisions_history),
                    ("backtest", lambda: _backtest(settings))):
        try:
            data[key] = fn()
        except Exception as e:   # 某一块失败不影响其他部分显示
            data[key] = None
            data.setdefault("errors", {})[key] = f"{type(e).__name__}: {e}"[:300]
    return data


# ---------------------------------------------------------------- HTTP 服务
def serve(settings: Settings, port: int = 8765, cache_seconds: int = 30) -> None:
    lock = threading.Lock()
    cache: dict = {"t": 0, "data": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):   # 安静
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/data":
                force = "refresh=1" in self.path
                with lock:
                    if force or time.time() - cache["t"] > cache_seconds:
                        cache["data"] = collect(settings)
                        cache["t"] = time.time()
                    body = json.dumps(cache["data"], ensure_ascii=False, default=str).encode()
                self._send(200, body, "application/json; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"仪表盘已启动: http://127.0.0.1:{port}   （Ctrl+C 退出）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
