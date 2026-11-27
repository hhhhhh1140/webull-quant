"""Three-profile paper trading arena for SOXL/SOXS dual-layer strategy comparison.

Profiles:
- plan_a:    Updated Plan A (IRX & TNX > SMA60 short gate) -> Webull UAT sandbox (shares base `uat` ledger)
- plan_b:    Plan B Defensive (Rate gate + 21:00-23:59 + no reentry) -> Futu US SIMULATE (`futu-sim`) or local paper
- plan_orig: Original 2026 1150%/919% baseline (no rate gate) -> Local real-time paper (`paper`)
"""
from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import yaml

from . import journal
from . import swing_live as SL
from .config import ROOT, Settings

PAPER_CONF_DIR = ROOT / "conf" / "paper"


@dataclass(frozen=True)
class ArenaProfile:
    key: str
    label: str
    summary: str
    config_file: Path
    default_mode: str
    env_suffix: str  # empty string means reuse base settings.environment (e.g. "uat")


PROFILES: tuple[ArenaProfile, ...] = (
    ArenaProfile(
        key="plan_a",
        label="A版 · 利率双开关",
        summary="IRX&TNX>SMA60 才做空 SOXS；21:00–04:00；允许二腿；+模式A暴跌抓底",
        config_file=PAPER_CONF_DIR / "plan_a_rate_gate.yaml",
        default_mode="webull",
        env_suffix="",
    ),
    ArenaProfile(
        key="plan_b",
        label="B版 · 防收割满仓",
        summary="IRX&TNX>SMA60 才做空 SOXS；仅 21:00–23:59；禁二腿；+模式A暴跌抓底",
        config_file=PAPER_CONF_DIR / "plan_b_defensive.yaml",
        default_mode="paper",
        env_suffix="plan_b",
    ),
    ArenaProfile(
        key="plan_orig",
        label="原版 · 2026 基准",
        summary="无利率开关（原 1150%/919% 版）；21:00–04:00；允许二腿；+模式A暴跌抓底",
        config_file=PAPER_CONF_DIR / "plan_orig_baseline.yaml",
        default_mode="paper",
        env_suffix="plan_orig",
    ),
)

PROFILE_MAP: dict[str, ArenaProfile] = {p.key: p for p in PROFILES}


def profile_env(base_env: str, profile: ArenaProfile) -> str:
    return f"{base_env}_{profile.env_suffix}" if profile.env_suffix else base_env


def load_profile_settings(base: Settings, profile: ArenaProfile) -> Settings:
    """Merge base config.yaml with profile override YAML and isolate state by environment key."""
    merged = json.loads(json.dumps(base.raw))
    if profile.config_file.exists():
        over = yaml.safe_load(profile.config_file.read_text(encoding="utf-8")) or {}
        SL._deep_merge(merged, over)
    return replace(base, raw=merged, environment=profile_env(base.environment, profile))


def _max_drawdown(values: list[float]) -> float:
    peak, dd = None, 0.0
    for v in values:
        peak = v if peak is None else max(peak, v)
        if peak and peak > 0:
            dd = min(dd, v / peak - 1.0)
    return dd


def _profile_snapshot(base: Settings, profile: ArenaProfile, prices: dict[str, float] | None = None,
                      now: datetime | None = None) -> dict:
    ps = load_profile_settings(base, profile)
    env = ps.environment
    cfg = SL.swing_cfg(ps)
    tcfg = cfg["tactical"]
    cap = float(tcfg["capital"])
    eqp = SL._equity_path(env)
    eq_data = json.loads(eqp.read_text(encoding="utf-8")) if eqp.exists() else {"capital": cap, "equity": cap, "history": []}
    booked = float(eq_data.get("equity", cap))
    history = eq_data.get("history") or []
    books = SL.Books(env).data
    core = SL._load_core_state(env)
    now_et = now or datetime.now(SL.ET)
    day = SL.trading_day(now_et)
    sp = SL._state_path(env, day)
    st = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}

    # Compute live unrealized PnL for any open core or tactical position
    unrealized = 0.0
    holding_parts: list[str] = []
    px_map = prices or {}

    core_sym = (tcfg.get("trend_switch") or {}).get("core_symbol", tcfg["long"])
    core_qty = float((books.get("core") or {}).get(core_sym, 0))
    if core_qty > 0:
        entry = float(core.entry or 0.0)
        cur_px = px_map.get(core_sym)
        if cur_px and entry > 0:
            unrealized += (cur_px - entry) * core_qty
        holding_parts.append(f"核心 {core_sym} {int(core_qty)}股" + (f"@{entry:.2f}" if entry else ""))

    for sym, qty in (books.get("tactical") or {}).items():
        if qty > 0:
            entry = float(st.get("entry") or 0.0) if st.get("symbol") == sym else 0.0
            cur_px = px_map.get(sym)
            if cur_px and entry > 0:
                unrealized += (cur_px - entry) * float(qty)
            holding_parts.append(f"战术 {sym} {int(qty)}股" + (f"@{entry:.2f}" if entry else ""))

    equity = round(booked + unrealized, 2)
    ret = (equity / cap - 1.0) if cap else 0.0
    dd = _max_drawdown([cap] + [float(h["equity"]) for h in history] + [equity])
    states = sorted(journal.STATE.glob(f"swing_tactical_{env}_*.json"))
    last_log = (st.get("log") or [])[-1] if st.get("log") else ""
    macro_cfg = tcfg.get("macro_short_gate") or {}
    crash_cfg = tcfg.get("crash_reversal") or {}
    return {
        "key": profile.key,
        "label": profile.label,
        "summary": profile.summary,
        "env": env,
        "default_mode": profile.default_mode,
        "capital": cap,
        "equity": equity,
        "realized": round(booked - cap, 2),
        "unrealized": round(unrealized, 2),
        "return_pct": round(ret, 5),
        "max_drawdown": round(dd, 5),
        "holding": " / ".join(holding_parts) or "空仓",
        "traded_days": len(history),
        "watched_days": len(states),
        "day": str(day),
        "trend_votes": st.get("trend_votes", core.last_votes),
        "macro_gate_enabled": bool(macro_cfg.get("enabled")),
        "macro_short_allowed": st.get("macro_short_allowed"),
        "macro_short_note": st.get("macro_short_note", ""),
        "crash_reversal_enabled": bool(crash_cfg.get("enabled")),
        "crash_symbol": st.get("crash_symbol", ""),
        "crash_active": bool(st.get("crash_active")),
        "crash_note": st.get("crash_note", ""),
        "gate": st.get("gate", ""),
        "last_log": last_log,
        "history": history[-120:],
    }


def arena_snapshot(base: Settings, prices: dict[str, float] | None = None, now: datetime | None = None) -> list[dict]:
    return [_profile_snapshot(base, p, prices=prices, now=now) for p in PROFILES]


def print_arena_status(base: Settings) -> int:
    rows = arena_snapshot(base)
    print(f"三套 SOXL/SOXS 模拟盘并排对比  ({datetime.now(SL.ET):%Y-%m-%d %H:%M} ET)\n")
    for r in rows:
        mg = "关" if not r["macro_gate_enabled"] else (
            "允许空" if r["macro_short_allowed"] is True else
            "禁空" if r["macro_short_allowed"] is False else "待评估"
        )
        cr = "关" if not r["crash_reversal_enabled"] else (
            f"持仓中({r['crash_symbol']})" if r["crash_active"] else
            f"唤醒({r['crash_symbol']})" if r["crash_symbol"] else "待机(≤-25%)"
        )
        print(f"【{r['label']}】 ({r['key']} | 账本 {r['env']} | 默认 {r['default_mode']})")
        print(f"  规则: {r['summary']}")
        print(f"  本金 ${r['capital']:,.0f}  权益 ${r['equity']:,.2f}  收益 {r['return_pct']:+.2%}"
              f"（已实现 ${r['realized']:+,.2f} / 浮动 ${r['unrealized']:+,.2f}）  最大回撤 {r['max_drawdown']:+.2%}")
        print(f"  持仓: {r['holding']}  |  利率门控: {mg}  |  模式A抓底: {cr}  |  已监控 {r['watched_days']} 日 / 结算 {r['traded_days']} 笔")
        if r["last_log"]:
            print(f"  最新: {r['last_log']}")
        print()
    return 0


def build_arena_runners(
    base: Settings,
    selected_keys: list[str],
    *,
    a_mode: str = "webull",
    b_mode: str = "futu-sim",
    orig_mode: str = "paper",
    confirm_live: bool = False,
) -> list[tuple[ArenaProfile, Settings, object, SL.Books]]:
    mode_map = {"plan_a": a_mode, "plan_b": b_mode, "plan_orig": orig_mode}
    runners = []
    for key in selected_keys:
        prof = PROFILE_MAP[key]
        ps = load_profile_settings(base, prof)
        cfg = SL.swing_cfg(ps)
        mode = mode_map.get(key, prof.default_mode)
        if mode == "futu-sim":
            try:
                broker = SL.make_broker(ps, "futu-sim", confirm_live, cfg["order"], label=prof.key)
            except Exception as exc:
                print(f"[arena:{prof.key}] 富途模拟盘初始化失败（{exc}），自动降级为本地实时模拟盘")
                broker = SL.make_broker(ps, "paper", confirm_live, cfg["order"], label=f"{prof.key}-fallback")
        else:
            broker = SL.make_broker(ps, mode, confirm_live, cfg["order"], label=prof.key)
        books = SL.Books(ps.environment)
        runners.append((prof, ps, broker, books))
    return runners


def run_arena_tick(
    runners: list[tuple[ArenaProfile, Settings, object, SL.Books]],
    md: SL.Market,
    now: datetime,
    *,
    day: date | None = None,
) -> dict[str, SL.TacState]:
    """Advance every selected paper profile by one poll step using a shared market feed."""
    target_day = day or SL.trading_day(now)
    states: dict[str, SL.TacState] = {}
    for prof, ps, broker, books in runners:
        st = SL._load_state(ps.environment, target_day)
        tcfg = SL.swing_cfg(ps)["tactical"]
        ts_cfg = tcfg.get("trend_switch") or {}
        if st.done and ts_cfg.get("enabled"):
            cst = SL._load_core_state(ps.environment)
            waiting_rebuy = st.trend_regime == 1 and st.trades == 0 and cst.stopped and st.day > cst.stopped_day
            if books.qty("core", ts_cfg.get("core_symbol", tcfg["long"])) or waiting_rebuy:
                st.done = False
        if not st.done:
            try:
                st = SL.tactical_step(ps, md, broker, books, st, now)
            except Exception as exc:
                st.log.append(f"{now.astimezone(SL.ET):%m-%d %H:%M} [{prof.key}] 出错，稍后重试：{type(exc).__name__}: {exc}")
                print(st.log[-1])
            SL._save_state(ps.environment, st)
        states[prof.key] = st
    return states


def run_arena_watch(
    base: Settings,
    md: SL.Market,
    selected_keys: list[str],
    now_fn: Callable[[], datetime],
    *,
    a_mode: str = "webull",
    b_mode: str = "futu-sim",
    orig_mode: str = "paper",
    confirm_live: bool = False,
    watch: bool = True,
    poll_seconds: int = 60,
    sleep_fn: Callable[[float], None] = _time.sleep,
) -> dict[str, SL.TacState]:
    """Run selected paper profiles in lockstep with a shared cached market feed."""
    shared_md = md if isinstance(md, SL.CachedMarket) else SL.CachedMarket(md)
    runners = build_arena_runners(
        base,
        selected_keys,
        a_mode=a_mode,
        b_mode=b_mode,
        orig_mode=orig_mode,
        confirm_live=confirm_live,
    )
    print(f"[paper-arena] 启动 {len(runners)} 套模拟盘：")
    for prof, ps, broker, _ in runners:
        print(f"  - {prof.label} ({prof.key}) -> 账本={ps.environment} 执行器={getattr(broker, 'name', type(broker).__name__)}")

    while True:
        now = now_fn()
        day, opens, closes = SL.next_market_window(now)
        if now < opens:
            if not watch:
                print(f"[paper-arena] 当前休市，下场夜盘开盘 {opens:%Y-%m-%d %H:%M} ET")
                return {}
            wait_s = max(1.0, opens.timestamp() - now.timestamp())
            print(f"[paper-arena] 休市中，等待 {day} 夜盘开盘（{opens:%m-%d %H:%M} ET）...")
            sleep_fn(wait_s)
            continue

        states = run_arena_tick(runners, shared_md, now, day=day)
        if not watch:
            return states
        if all(st.done for st in states.values()):
            any_crash_enabled = any(
                bool((SL.swing_cfg(ps)["tactical"].get("crash_reversal") or {}).get("enabled"))
                for _, ps, _, _ in runners
            )
            win = SL.market_window(day)
            regular_close = win[1] if win else closes
            if any_crash_enabled and now < closes:
                if now < regular_close:
                    sleep_fn(max(1.0, min(closes.timestamp(), regular_close.timestamp() + 5.0) - now_fn().timestamp()))
                    continue
                nxt_day = SL.next_trading_day(day)
                nxt_states = run_arena_tick(runners, shared_md, now, day=nxt_day)
                if any(st.crash_symbol and not st.done for st in nxt_states.values()):
                    sleep_fn(max(1.0, min(float(poll_seconds), closes.timestamp() - now_fn().timestamp())))
                    continue
            remaining = closes.timestamp() - now_fn().timestamp()
            if remaining > 0:
                sleep_fn(remaining)
            continue
        sleep_fn(max(1.0, min(float(poll_seconds), closes.timestamp() - now_fn().timestamp())))
