"""用已经收盘的 M5 在本地模拟成交，按美东交易日保存独立账本。"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import journal, scalp_data
from .scalp import ScalpParams, run_session

ET = ZoneInfo("America/New_York")
STEP = pd.Timedelta(minutes=5)


class _QuoteError(RuntimeError):
    """SDK 异常可能包含请求信息，仅输出异常类型。"""


def _fingerprint(long_bars, short_bars) -> str:
    """上一次看到的历史不能被供应商后来修订的 OHLCV 悄悄覆盖。"""
    digest = hashlib.sha256()
    for bars in (long_bars, short_bars):
        prices = bars[["open", "high", "low", "close", "volume"]].astype(float)
        digest.update(pd.util.hash_pandas_object(prices, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _save(path: Path, book: dict) -> None:
    output = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    tmp = Path(output.name)
    try:
        with output:
            json.dump(book, output, ensure_ascii=False, allow_nan=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _exclude(record: dict, reason: str) -> None:
    record.update(status="excluded", reason=reason)


def _describe(day: str, record: dict) -> str:
    state = record["state"]
    result = (f"{day} ET  已处理 {record['last_bar']} 开始的 K 线  "
              f"已平仓 {len(record['trades'])} 笔  已实现 ${state['pnl']:+.2f}  "
              f"持仓 {state['symbol'] or '空仓'} {state['qty']} 股  可用现金 ${state['cash']:.2f}  "
              f"待执行 {state['pending'] or '无'}")
    if record["status"] == "excluded":
        return (f"{result}  当日记录不完整（{record['reason']}）。"
                "次日只结转可用现金，未结持仓继续单独记录。")
    return result + ("  当日已结束" if record["status"] == "closed" else "")


def _advance(book: dict, fetch_pair, params: ScalpParams, now: pd.Timestamp) -> str:
    today = now.date()
    day = today.isoformat()
    days = book["days"]
    older = max((key for key in days if key < day), default=None)
    if older and days[older]["status"] == "active":
        _exclude(days[older], "跨日时缺少前一交易日收盘数据")
    previous_notice = (_describe(older, days[older]) + "\n"
                       if older and days[older]["status"] == "excluded" else "")
    record = days.get(day)
    if record and record["status"] != "active":
        return _describe(day, record)
    bounds = scalp_data.session_bounds(today)
    if not bounds or now < bounds[0] + STEP:
        return previous_notice + f"{day} ET 等待常规时段首根已收盘 M5"
    close = bounds[1]
    if record and min(now, close) >= pd.Timestamp(record["last_bar"]) + 3 * STEP:
        _exclude(record, "错过连续的已收盘 M5，未模拟离线期间的成交")
        return _describe(day, record)
    if now >= close and not record:
        return previous_notice + f"{day} ET 已收盘，等待下一个交易日"

    # 每次只向行情接口读数据。as_of 取请求前的时间，防止请求跨过 K 线边界。
    try:
        long_bars, short_bars = fetch_pair()
    except Exception as exc:
        raise _QuoteError(f"Webull M5 行情读取失败（{type(exc).__name__}）") from None
    current = next(((a, b) for session_day, a, b, _ in
                    scalp_data.sessions(long_bars, short_bars, as_of=now)
                    if session_day == today), None)
    if current is None:
        status = "等待两只 ETF 从开盘到当前的完整 M5"
        return (previous_notice + (_describe(day, record) + "  " if record else "") + status)
    a, b = current
    last = a.index[-1]
    if record:
        prior = pd.Timestamp(record["last_bar"])
        if prior not in a.index or _fingerprint(a.loc[:prior], b.loc[:prior]) != record["fingerprint"]:
            _exclude(record, "行情源修改了此前已经入账的 K 线")
            return _describe(day, record)
        if prior == last:
            return _describe(day, record)
        if prior + STEP != last:
            _exclude(record, "行情时间不连续")
            return _describe(day, record)
    else:
        # 会话从观察时刻开始，之前的 K 线仅用于指标预热。
        # 仅结转最近会话的现金。已平仓盈亏可用，未观测到卖出的持仓继续隔离。
        capital = days[older]["state"]["cash"] if older else params.capital
        if capital <= 0:
            return f"{day} ET 模拟本金已耗尽"
        record = {"status": "active", "trade_from": now.isoformat(), "opening_capital": capital}
    trades, state = run_session(a, b, replace(params, capital=record["opening_capital"]),
                                close, trade_from=pd.Timestamp(record["trade_from"]))
    record.update(trades=trades, state=state, last_bar=last.isoformat(), fingerprint=_fingerprint(a, b))
    days[day] = record
    if last + STEP == close:
        if state["symbol"] or state["pending"]:
            _exclude(record, "收盘时仍有未结持仓或待执行指令")
        else:
            record["status"] = "closed"
    return previous_notice + _describe(day, record)


def _poll_once(fetch_pair, params: ScalpParams, path: Path, clock) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 锁覆盖取行情及账本更新，两个同时运行的轮询不会倒序提交数据。
    with path.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        book = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
                else {"params": asdict(params), "days": {}})
        if book["params"] != asdict(params):
            raise ValueError("模拟账本参数与本次运行不符")
        before = json.dumps(book["days"], sort_keys=True)
        try:
            message = _advance(book, fetch_pair, params, pd.Timestamp(clock()).tz_convert(ET))
        except _QuoteError:
            # 即便当日行情失败，也持久化上一交易日的中断记录。
            if before != json.dumps(book["days"], sort_keys=True):
                _save(path, book)
            raise
        if before != json.dumps(book["days"], sort_keys=True):
            _save(path, book)
        return message


def run_paper(settings, params: ScalpParams, watch: bool = False) -> int:
    """单次轮询或每 30 秒观察一次，在 state 下按行情环境、策略参数隔离。"""
    from .webull_client import WebullClient

    client = None

    def fetch_pair():
        nonlocal client
        if client is None:
            client = WebullClient(settings)
        return scalp_data.latest_pair(client)

    config = json.dumps(asdict(params), sort_keys=True).encode()
    key = hashlib.sha256(config).hexdigest()[:12]
    path = journal.STATE / f"scalp_paper_{settings.environment}_{key}.json"
    print(f"本地 M5 模拟交易账本 {path}", flush=True)
    previous = None
    try:
        while True:
            try:
                message = _poll_once(fetch_pair, params, path, lambda: datetime.now(ET))
            except _QuoteError as exc:
                client = None
                message = str(exc)
                if not watch:
                    print(message, flush=True)
                    return 1
            if message != previous:
                print(message, flush=True)
                previous = message
            if not watch:
                return 0
            time.sleep(30)
    except KeyboardInterrupt:
        return 0
