"""事件日历：宏观数据 / 美联储 / 财报。

- 宏观事件写在项目根目录的 events.yaml（Claude 审核时用网上搜索维护，附市场预期）
- 财报日期从 Webull 接口自动获取（标的池里的股票）
- execute 时根据当前时间判断：重大事件公布前，新开仓位按系数缩小；财报前几天不买该股票
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import yaml

from .config import ROOT

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
EVENTS_FILE = ROOT / "events.yaml"
EARNINGS_CACHE = ROOT / "state" / "earnings.json"

# 财报没有具体时间时，保守地按收盘后处理
NAMED_TIMES = {"BMO": time(8, 0), "AMC": time(16, 5), "": time(16, 5)}


def event_dt(ev: dict) -> datetime:
    """事件公布时间（美东）。time_et 可以是 "08:30" / "14:00" / "BMO" / "AMC"。"""
    d = date.fromisoformat(str(ev["date"]))
    t = str(ev.get("time_et") or "").strip().upper()
    tm = NAMED_TIMES.get(t) or time.fromisoformat(t)
    return datetime.combine(d, tm, tzinfo=ET)


def load_macro() -> list[dict]:
    if not EVENTS_FILE.exists():
        return []
    data = yaml.safe_load(EVENTS_FILE.read_text(encoding="utf-8")) or {}
    out = []
    for ev in data.get("events", []) or []:
        try:
            event_dt(ev)
        except Exception as e:
            log.warning("events.yaml 里有一条格式不对，跳过: %s (%s)", ev, e)
            continue
        out.append({"type": "macro", "importance": "high", **ev})
    return out


def refresh_earnings(client, symbols: list[str]) -> list[dict]:
    """从 Webull 拉取标的池的下一次财报日期，缓存到 state/earnings.json。"""
    from .webull_client import _json, _unwrap_list
    out = []
    today = date.today()
    for s in symbols:
        try:
            rows = _unwrap_list(_json(client.data.fundamentals.get_earnings_calendar(s)))
        except Exception as e:
            log.info("%s 财报日历不可用: %s", s, e)
            continue
        future = [r for r in rows if r.get("expected_publish_date")
                  and date.fromisoformat(r["expected_publish_date"]) >= today - timedelta(days=1)]
        if future:
            r = min(future, key=lambda x: x["expected_publish_date"])
            out.append({"type": "earnings", "importance": "high", "symbol": s, "date": r["expected_publish_date"],
                        "time_et": "AMC", "name": f"{s} 财报",
                        "consensus": {"eps": r.get("eps_est"), "revenue": r.get("rev_est")},
                        "actual": {"eps": r.get("eps_actual"), "revenue": r.get("rev_actual")}
                        if r.get("eps_actual") else None})
    EARNINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    EARNINGS_CACHE.write_text(json.dumps({"updated": str(today), "events": out}, ensure_ascii=False, indent=1))
    return out


def load_earnings() -> list[dict]:
    if not EARNINGS_CACHE.exists():
        return []
    return json.loads(EARNINGS_CACHE.read_text()).get("events", [])


def all_events() -> list[dict]:
    return sorted(load_macro() + load_earnings(), key=lambda e: event_dt(e))


def upcoming(events: list[dict], now: datetime, days: int = 7, past_days: int = 2) -> list[dict]:
    """未来 days 天和过去 past_days 天的事件（过去的用于事后复盘）。"""
    lo, hi = now - timedelta(days=past_days), now + timedelta(days=days)
    out = []
    for ev in events:
        dt = event_dt(ev)
        if lo <= dt <= hi:
            out.append({**ev, "datetime_et": dt.isoformat(), "released": dt <= now,
                        "hours_until": round((dt - now).total_seconds() / 3600, 1)})
    return out


def restrictions(events: list[dict], now: datetime, cfg: dict) -> dict:
    """根据当前时间算出开仓限制。

    返回 {"size_factor": float, "reasons": [...], "blocked": {symbol: reason}}
    - 重大宏观事件在 lookahead_hours 内公布（还没公布）→ 新开仓位乘 pre_event_size_factor
    - 某只股票在 earnings_blackout_days 个交易日内发财报 → 不买这只
    """
    out = {"size_factor": 1.0, "reasons": [], "blocked": {}}
    if not cfg.get("enabled", True):
        return out
    look = timedelta(hours=cfg.get("lookahead_hours", 24))
    factor = float(cfg.get("pre_event_size_factor", 0.5))
    for ev in events:
        dt = event_dt(ev)
        if ev.get("type") == "earnings":
            continue
        if ev.get("importance", "high") == "high" and now < dt <= now + look:
            out["size_factor"] = min(out["size_factor"], factor)
            out["reasons"].append(f"{ev.get('name')}（{dt.strftime('%m-%d %H:%M')} ET）公布前，新开仓位 ×{factor}")
    blackout = int(cfg.get("earnings_blackout_days", 3))
    import numpy as np
    for ev in events:
        if ev.get("type") != "earnings" or not ev.get("symbol"):
            continue
        dt = event_dt(ev)
        if dt < now:
            continue
        bdays = int(np.busday_count(now.date(), dt.date()))
        if bdays <= blackout:
            out["blocked"][ev["symbol"]] = f"{ev['date']} 发财报（{bdays} 个交易日内），财报前不开新仓"
    return out
