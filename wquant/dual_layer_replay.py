"""Research-only SOXL/SOXS replay. Equity starts at $3,000 in each window.

Input: Yahoo daily OHLC/events and KOSPI intraday, plus raw Webull ETF bars.
Run with python -m wquant.dual_layer_replay INPUT_DIR OUTPUT_DIR.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
import json
from math import ceil, floor
from pathlib import Path

import pandas as pd

from .config import ROOT
from .swing import ensemble_votes

ET, KST = "America/New_York", "Asia/Seoul"
TICKERS = ("SOXL", "SOXS")
DAY = timedelta(days=1)
CAPITAL = 3_000
END = date(2026, 11, 24)
OVERNIGHT_LAUNCH = date(2024, 11, 12)
KRX_LATE_CLOSE = {date(2026, 11, 19): "16:35"}


def local(day, hhmm, zone=ET):
    return pd.Timestamp(f"{day} {hhmm}", tz=zone)


def read_prices(path):
    d = pd.read_csv(path, index_col=0)
    d.index = pd.to_datetime(d.index, utc=True, format="mixed")
    return d.loc[~d.index.duplicated(keep="last")].sort_index()


def read_actions(folder, sym, name):
    file = folder / f"{sym}_{name}.csv"
    if not file.exists() or file.stat().st_size < 5:
        return {}
    d = pd.read_csv(file)
    dates = pd.to_datetime(d.date, unit="s", utc=True).dt.tz_convert(ET).dt.date
    if name == "splits":
        return dict(zip(dates, d.numerator / d.denominator))
    return dict(zip(dates, d.amount))


def daily_data(folder, sym):
    d = read_prices(folder / f"{sym}.csv")
    d.index = d.index.tz_convert(ET).date
    return d


def native_daily(d, splits):
    d = d.copy()
    cols = ["open", "high", "low", "close"]
    for ex, ratio in splits.items():
        d.loc[d.index < ex, cols] *= ratio
    return d


class Korea:
    def __init__(self, folder):
        daily = read_prices(folder / "KOSPI.csv")["close"]
        daily.index = daily.index.tz_convert(KST).date
        previous = daily.shift()
        self.close_times = {d: local(d, KRX_LATE_CLOSE.get(d, "15:35"), KST) for d in previous.index}
        self.moves = {}
        for interval, filename, candidates in (
            (60, "KOSPI_hour.csv", [ROOT / "data/intraday/KOSPI_1h.csv"]),
            (5, "KOSPI_five.csv", [ROOT / "data/intraday/KOSPI_5m.csv"]),
            (1, "KOSPI_minute.csv", []),
        ):
            series = []
            snapshot = folder / f"KOSPI_{interval}m_closed.csv.gz"
            if snapshot.exists():
                candidates = [snapshot]
            for p in candidates:
                if p.exists():
                    series.append(read_prices(p).close)  # project cache stores bar end
            f = folder / filename
            sources = [f] if f.exists() else sorted(folder.glob("KOSPI_1m_*.csv")) if interval == 1 else []
            for src in ([] if snapshot.exists() else sources):
                s = read_prices(src).close
                s = s[s.index.second == 0]
                s.index += pd.Timedelta(minutes=interval)
                series.append(s)
            if not series:
                continue
            s = pd.concat(series).sort_index(kind="stable")
            s = s.loc[~s.index.duplicated(keep="last")]
            kst = s.index.tz_convert(KST)
            for day, values in s.groupby(kst.date):
                if day not in previous or pd.isna(previous[day]):
                    continue
                last = self.close_times[day]
                start = local(day, "10:00" if day in KRX_LATE_CLOSE else "09:00", KST)
                values = values[(values.index > start) & (values.index < last)]
                # A single partial day of five-minute bars would displace much more hourly data.
                if len(values) >= {1: 180, 5: 30, 60: 3}[interval]:
                    values = values / previous[day] - 1
                    self.moves[day] = (list(values.index), list(values), f"KOSPI_{interval}m")
        self.closes = (daily / previous - 1).dropna().to_dict()


    def lookup(self, ts):
        day = ts.tz_convert(KST).date()
        if day not in self.closes:
            return None, "KRX_closed"
        if ts >= self.close_times[day]:
            return self.closes[day], "KOSPI_close"
        if day in self.moves:
            stamps, values, source = self.moves[day]
            i = bisect_right(stamps, ts) - 1
            if i >= 0:
                return values[i], source
        return None, "KOSPI_wait"


@dataclass(frozen=True, slots=True)
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float
    session: str


class Intraday:
    def __init__(self, folder, sessions, daily, splits):
        self.by_step = {}
        self.rescaled = Counter()
        session_days = set(sessions)
        for minutes in (1, 5):
            book = {}
            for symbol in TICKERS:
                if minutes == 1:
                    archive = folder / f"{symbol}_M1.csv.gz"
                    paths = [archive] if archive.exists() else sorted(folder.glob(f"{symbol}_m*.csv"))
                else:
                    archive = folder / f"{symbol}_M5.csv.gz"
                    paths = [archive] if archive.exists() else [ROOT / f"data/intraday/{symbol}_M5.csv"]
                pieces = [read_prices(p) for p in paths if p.exists()]
                if not pieces:
                    continue
                d = pd.concat(pieces).sort_index(kind="stable")
                d = d.loc[~d.index.duplicated(keep="last")]
                d.index = d.index.tz_convert(ET)
                for stamp, *fields in d[["open", "high", "low", "close", "volume", "session"]].itertuples():
                    day = stamp.date() + (DAY if stamp.hour >= 20 else timedelta())
                    if day in session_days:
                        book.setdefault(day, {}).setdefault(stamp, {})[symbol] = Bar(*fields)
            self.by_step[minutes] = book
        self.step = {}
        self.audit = Counter()
        self.close_diff = {1: [], 5: []}
        for day in sessions:
            for minutes in (1, 5):
                records = self.by_step[minutes].get(day, {})
                if self.complete(records, minutes):
                    normalized = self.normalize(records, day, daily, splits, minutes)
                    if not normalized:
                        continue
                    self.step[day] = minutes
                    for symbol in TICKERS:
                        close = max((ts, bars[symbol].close) for ts, bars in records.items()
                                    if symbol in bars and bars[symbol].session == "RTH")[1]
                        self.close_diff[minutes].append(abs(close / daily[symbol].at[day, "close"] - 1))
                    break
            self.audit[self.step.get(day, 0)] += 1

    def normalize(self, records, day, daily, splits, minutes):
        factors = {}
        for sym in TICKERS:
            _, last = max((ts, b[sym].close) for ts, b in records.items()
                          if sym in b and b[sym].session == "RTH")
            ratio = last / daily[sym].at[day, "close"]
            factor = 1.0
            for ex, split in splits[sym].items():
                if day < ex:
                    factor /= split
            if abs(ratio - 1) < .015:
                factors[sym] = 1.
            elif abs(ratio / factor - 1) < .015:
                factors[sym] = factor
            else:
                return False
        for sym, factor in factors.items():
            if factor != 1:
                self.rescaled[f"M{minutes}_{sym}"] += 1
                for bars in records.values():
                    if sym in bars:
                        b = bars[sym]
                        bars[sym] = Bar(b.open / factor, b.high / factor, b.low / factor,
                                        b.close / factor, b.volume * factor, b.session)
        return True

    @staticmethod
    def complete(records, minutes):
        counts = Counter((sym, b.session) for bars in records.values() for sym, b in bars.items())
        last = {}
        night = {sym: [] for sym in TICKERS}
        for ts, bars in records.items():
            for sym, bar in bars.items():
                if bar.session == "RTH":
                    last[sym] = max(ts, last.get(sym, ts))
                elif bar.session == "OVN":
                    night[sym].append(ts)
        if len(last) < 2 or len({ts.hour for ts in last.values()}) != 1:
            return False
        early = next(iter(last.values())).hour == 12
        need = ceil((210 if early else 390) * .9 / minutes)
        session_day = next(iter(last.values())).date()
        scan_start = local(session_day - DAY, "21:00")
        scan_end = local(session_day, "04:00")
        return all(counts[s, "RTH"] >= need and counts[s, "OVN"] >= 300 / minutes
                   and last[s].minute >= 60 - minutes and min(night[s]) <= scan_start
                   and max(night[s]) >= scan_end - pd.Timedelta(minutes=5) for s in TICKERS)

    def day(self, day):
        step = self.step.get(day, 0)
        return step, self.by_step[step].get(day, {}) if step else {}

    def afterhours(self, day, step):
        stamps = self.by_step[step].get(day, {})
        return [v["SOXL"] for t, v in stamps.items() if t.hour >= 16 and v.get("SOXL") is not None
                and v["SOXL"].session == "ATH"]


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: int
    price: float
    layer: str
    peak: float


@dataclass(slots=True)
class Order:
    symbol: str
    action: str
    layer: str
    reason: str
    limit: float
    quantity: int
    submitted: pd.Timestamp
    trigger_basis: float | None


class Replay:
    def __init__(self, offset):
        self.offset = offset
        self.cash = float(CAPITAL)
        self.position = None
        self.order = None
        self.core_stop = None
        self.trades = []
        self.stats = Counter()
        self.pnl = Counter()
        self.equity = []

    def submit(self, symbol, action, layer, reason, price, t, *, trigger_basis=None):
        sign = 1 if action == "BUY" else -1
        limit = max(.01, round(price * (1 + sign * self.offset), 2))
        qty = floor(self.cash / limit) if action == "BUY" else self.position.quantity
        if qty:
            self.order = Order(symbol, action, layer, reason, limit, qty, t, trigger_basis)

    def fill(self, t, price):
        order = self.order
        self.order = None
        qty = order.quantity
        pnl = 0.0
        if order.action == "BUY":
            self.cash -= price * qty
            self.position = Position(order.symbol, qty, price, order.layer, price)
        else:
            pnl = (price - self.position.price) * qty
            self.cash += price * qty
            self.position = None
            self.pnl[order.layer] += pnl
        self.stats[order.reason] += 1
        self.trades.append(dict(session=self.day.isoformat(), submit=order.submitted.isoformat(),
                                fill=t.isoformat(), side=order.action, symbol=order.symbol, layer=order.layer,
                                reason=order.reason, limit=order.limit, price=price, quantity=qty,
                                pnl=pnl, cash=self.cash, fidelity=self.fidelity, trigger_basis=order.trigger_basis))
        if order.action == "SELL" and order.layer == "tactical":
            self.pnl[order.symbol] += pnl
        if order.reason == "tactical_entry":
            self.stats[f"entry_{order.symbol}"] += 1
        if order.reason == "core_stop":
            self.core_stop = self.day
        elif order.action == "BUY" and order.layer == "core":
            self.core_stop = None
        elif order.reason in ("tactical_stop", "tactical_trail"):
            self.tactical_low, self.tactical_exit = price, t
        elif order.action == "BUY" and order.layer == "tactical":
            self.tactical_legs += 1
            self.tactical_symbol = order.symbol
            self.tactical_low = None
            self.tactical_exit = None

    def try_fill(self, t, bar):
        o = self.order
        if bar is None or bar.volume < o.quantity:
            return
        if (o.action == "BUY" and bar.low <= o.limit) or (o.action == "SELL" and bar.high >= o.limit):
            self.fill(t, o.limit)

    def mark(self, day, price, votes):
        value = self.cash + (self.position.quantity * price if self.position else 0)
        self.equity.append((day, value, self.fidelity, int(votes)))

    def reset_day(self, day, step, splits, distributions, opens):
        if self.order:
            self.stats[f"expired_{self.order.reason}"] += 1
            self.order = None
        self.day = day
        self.fidelity = f"M{step}" if step else "D"
        self.stats[f"days_{self.fidelity}"] += 1
        self.tactical_legs, self.tactical_low, self.tactical_symbol = 0, None, None
        self.tactical_exit = None
        if self.position:
            p = self.position
            ratio = splits[p.symbol].get(day, 1)
            if ratio != 1:
                new = round(p.quantity * ratio, 8)
                self.cash += (new % 1) * opens[p.symbol]
                p.quantity = floor(new)
                p.price /= ratio
                p.peak /= ratio
                if not p.quantity:
                    self.position = None
            div = distributions[p.symbol].get(day, 0)
            dividend = div * p.quantity
            self.cash += dividend
            self.pnl[f"{p.layer}_dividends"] += dividend

    def approximate(self, day, votes, bar, prices):
        """Missing minute data: regular-hours core only, with conservative OHLC ordering."""
        opening, exit_t = local(day, "09:30"), local(day, "16:00")
        if self.position and (votes < 2 or self.position.layer == "tactical"):
            p = self.position
            self.submit(p.symbol, "SELL", p.layer, "trend_exit" if p.layer == "core" else "late_flat",
                        prices[p.symbol].open, opening)
            self.fill(opening, self.order.limit)
        if not self.position and votes >= 2 and self.core_stop is None:
            self.submit("SOXL", "BUY", "core", "core_entry", bar.open, opening)
            if self.order:
                self.fill(opening, self.order.limit)
        if self.position:
            basis = self.position.price
            stop = basis * .88
            if bar.low <= stop:
                self.submit("SOXL", "SELL", "core", "core_stop", min(bar.open, stop), exit_t, trigger_basis=basis)
                self.fill(exit_t, self.order.limit)
        elif votes >= 2 and self.core_stop is not None and self.core_stop < day and bar.close >= bar.low * 1.05:
            # An end-of-day close confirms this session's low came before the rebound.
            self.submit("SOXL", "BUY", "core", "core_rebuy", bar.close, exit_t, trigger_basis=bar.low)
            if self.order:
                self.fill(exit_t, self.order.limit)

    def replay_day(self, day, preparation, votes, step, records, prev, korea):
        length = pd.Timedelta(minutes=step)
        start, stop_scan = local(day - DAY, "21:00"), local(day, "04:00")
        regular, reentry = local(day, "09:30"), local(day, "15:00")
        # The 12:55 cutoff is used if the verified session ends at 13:00.
        last_rth = max(t for t, b in records.items() if "SOXL" in b and b["SOXL"].session == "RTH")
        flatten = local(day, "12:55" if last_rth.hour == 12 else "15:55")
        previous = {"SOXL": prev}
        path_low = min([prev] + [b.low for b in preparation])
        path_high = max([prev] + [b.high for b in preparation])
        step_ends = {t + length: b for t, b in records.items()}
        scan_ticks = pd.date_range(start, stop_scan, freq=length)
        times = sorted(set(records) | set(step_ends) | set(scan_ticks) | {regular, flatten})
        open_attempted = False
        last_overnight = None
        regular_low = float("inf")
        allowed_target = {"SOXS"} if votes <= -2 else {"SOXL"} if votes >= 2 or self.core_stop else set(TICKERS)
        for t in times:
            new = records.get(t, {})
            closed = step_ends.get(t, {})
            for symbol, bar in closed.items():
                previous[symbol] = bar.close
            long = closed.get("SOXL")
            if long:
                if long.session == "OVN":
                    last_overnight = long.close
                path_low = min(path_low, long.low)
                path_high = max(path_high, long.high)
                if long.session == "RTH":
                    regular_low = min(regular_low, long.low)
            p = self.position
            if p and p.layer == "tactical" and t >= flatten:
                if self.order is None or self.order.reason != "flatten":
                    self.order = None
                    price = previous.get(p.symbol)
                    if price is not None:
                        self.submit(p.symbol, "SELL", p.layer, "flatten", price, t)
            elif t >= flatten and self.order and self.order.layer == "tactical" and self.order.action == "BUY":
                self.stats[f"cancelled_{self.order.reason}"] += 1
                self.order = None
            if not self.order and p:
                b = closed.get(p.symbol)
                if p.layer == "core" and votes < 2:
                    current = new.get(p.symbol)
                    if current is not None or b is not None:
                        self.submit(p.symbol, "SELL", p.layer, "trend_exit",
                                    b.close if b else current.open, t)
                elif p.layer == "core" and b and b.low <= p.price * .88:
                    self.submit(p.symbol, "SELL", p.layer, "core_stop", min(b.open, p.price * .88, b.close), t,
                                trigger_basis=p.price)
                elif p.layer == "tactical" and b and t < flatten:
                    stop = p.price * (.94 if self.tactical_legs == 1 else .88)
                    prior_trail = p.peak * .93
                    p.peak = max(p.peak, b.high)
                    if self.tactical_legs == 1 and prior_trail > stop and b.low <= prior_trail:
                        self.submit(p.symbol, "SELL", p.layer, "tactical_trail", min(b.open, prior_trail, b.close), t)
                    elif b.low <= stop:
                        self.submit(p.symbol, "SELL", p.layer, "tactical_stop", min(b.open, stop, b.close), t,
                                    trigger_basis=p.price)
                    elif self.tactical_legs == 1 and b.close <= p.peak * .93:
                        self.submit(p.symbol, "SELL", p.layer, "tactical_trail", b.close, t)
            elif not self.order and not p and t < flatten:
                if votes >= 2 and self.tactical_legs == 0 and regular <= t and not open_attempted:
                    if self.core_stop is None:
                        b = new.get("SOXL")
                        if b and b.session == "RTH":
                            self.submit("SOXL", "BUY", "core", "core_entry", b.open, t)
                            open_attempted = True
                    elif self.core_stop < day and long and long.session == "RTH" and long.close >= regular_low * 1.05:
                        self.submit("SOXL", "BUY", "core", "core_rebuy", long.close, t, trigger_basis=regular_low)
                        open_attempted = True
                if (not self.order and self.tactical_legs == 1 and self.tactical_low is not None and t < reentry):
                    b = closed.get(self.tactical_symbol)
                    if b:
                        # With bar data the sell may occur after the low in its execution bar.
                        post_exit_low = b.low if t - length > self.tactical_exit else b.close
                        self.tactical_low = min(self.tactical_low, post_exit_low)
                        if b.close >= self.tactical_low * 1.03:
                            self.submit(self.tactical_symbol, "BUY", "tactical", "tactical_rebuy", b.close, t,
                                        trigger_basis=self.tactical_low)
                if (not self.order and self.tactical_legs == 0 and start <= t <= stop_scan
                        and (votes < 2 or (self.core_stop is not None and self.core_stop < day))
                        and last_overnight is not None):
                    price = last_overnight
                    south, source = korea.lookup(t)
                    if south is not None:
                        direction = None
                        if price >= prev * 1.03 and south >= .01 and price > path_high * .94:
                            direction = "SOXL"
                        elif price <= prev * .97 and south <= -.01 and price < path_low * 1.06:
                            direction = "SOXS"
                        if direction in allowed_target and direction in previous:
                            self.submit(direction, "BUY", "tactical", "tactical_entry", previous[direction], t)
                            if self.order:
                                self.stats[f"signal_{source}"] += 1
            if self.order:
                is_flatten = self.order.reason == "flatten"
                self.try_fill(t, new.get(self.order.symbol))
                if is_flatten and not self.order and t >= last_rth + length:
                    self.stats["flatten_after_rth"] += 1
        if self.position and self.position.layer == "tactical":
            self.stats["unfilled_forced_exits"] += 1
        return previous


def summarize(run, benchmark):
    frame = pd.DataFrame(run.equity, columns=["date", "equity", "fidelity", "votes"]).set_index("date")
    base = pd.concat([pd.Series([CAPITAL]), frame.equity], ignore_index=True)
    ret = base.pct_change().dropna()
    sells = [t for t in run.trades if t["side"] == "SELL"]
    gains = sum(max(0, t["pnl"]) for t in sells)
    losses = sum(max(0, -t["pnl"]) for t in sells)
    performance = dict(start=str(frame.index[0]), end=str(frame.index[-1]), days=len(frame),
                       equity=float(base.iloc[-1]), total=float(base.iloc[-1] / CAPITAL - 1),
                       cagr=float((base.iloc[-1] / CAPITAL) ** (365.2425 / max(1, (frame.index[-1] - frame.index[0]).days)) - 1),
                       max_drawdown=float((base / base.cummax() - 1).min()),
                       sharpe=float(ret.mean() / ret.std() * 252 ** .5) if ret.std() else 0.,
                       benchmark=float(benchmark.at[frame.index[-1]] / benchmark.at[frame.index[0]] - 1),
                       closes=len(sells), win_rate=sum(t["pnl"] > 0 for t in sells) / len(sells) if sells else None,
                       profit_factor=gains / losses if losses else None,
                       stats=dict(run.stats), pnl=dict(run.pnl))
    return performance, frame


def monthly_breakdown(run, curve):
    trades = pd.DataFrame(run.trades)
    tactical = trades.loc[(trades.side == "SELL") & (trades.layer == "tactical")]
    tactical = tactical.groupby(tactical.session.str[:7]).pnl.sum()
    monthly = []
    start = CAPITAL
    for month, rows in curve.groupby([d.isoformat()[:7] for d in curve.index]):
        end = rows.equity.iloc[-1]
        path = pd.concat([pd.Series([start]), rows.equity], ignore_index=True)
        realized_tactics = tactical.get(month, 0.)
        monthly.append(dict(month=month, start_date=rows.index[0], end_date=rows.index[-1],
                            opening_equity=start, closing_equity=end, net_pnl=end-start,
                            monthly_return=end/start-1, max_drawdown=(path/path.cummax()-1).min(),
                            core_pnl=end-start-realized_tactics, tactical_pnl=realized_tactics,
                            m1_days=int((rows.fidelity=="M1").sum()), m5_days=int((rows.fidelity=="M5").sum())))
        start = end
    return pd.DataFrame(monthly)


def main(source, output):
    data = {sym: daily_data(source, sym) for sym in ("SOXX", *TICKERS)}
    for sym in data:
        data[sym] = data[sym].loc[:END]
    votes = ensemble_votes(data["SOXX"]).shift()
    splits = {s: read_actions(source, s, "splits") for s in TICKERS}
    cash_div = {s: read_actions(source, s, "dividends") for s in TICKERS}
    for s in TICKERS:
        # Yahoo historical cash dividends are adjusted for later splits, as are daily OHLCs.
        cash_div[s] = {d: amount * pd.Series([r for ex, r in splits[s].items() if ex > d], dtype=float).prod()
                       for d, amount in cash_div[s].items()}
    raw = {sym: native_daily(data[sym], splits[sym]) for sym in TICKERS}
    days = sorted(set.intersection(*(set(df.index) for df in data.values())))
    minute = Intraday(source, [d for d in days if d >= OVERNIGHT_LAUNCH], raw, splits)
    korea = Korea(source)
    tradable = [d for d, resolution in minute.step.items() if resolution and d <= END]
    windows = {
        "2020_2026": date(2020, 1, 1),
        "2026_ytd": date(2026, 1, 1),
        "minute_covered": min(tradable),
    }
    first_m1 = min((d for d, res in minute.step.items() if res == 1), default=None)
    if first_m1 is not None:
        windows["m1_covered"] = first_m1
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    annual = []
    for key, start in windows.items():
        for variant in ("hybrid", "core_only"):
            if variant == "core_only" and key not in ("2020_2026", "2026_ytd"):
                continue
            run = Replay(.003)
            for i, d in enumerate(days):
                if d < start:
                    continue
                yesterday = days[i - 1]
                bars = {s: raw[s].loc[d] for s in TICKERS}
                prev = float(raw["SOXL"].at[yesterday, "close"]) / splits["SOXL"].get(d, 1)
                vote = votes.at[d]
                step, records = minute.day(d)
                if key == "m1_covered" and step != 1:
                    step = 0
                run.reset_day(d, step, splits, cash_div, {s: bars[s].open for s in TICKERS})
                if vote < 2:
                    run.core_stop = None
                if step:
                    preparation = minute.afterhours(yesterday, step)
                    k = korea if variant == "hybrid" else NoKorea()
                    run.replay_day(d, preparation, vote, step, records, prev, k)
                else:
                    run.stats["pre_ovn_launch" if d < OVERNIGHT_LAUNCH else "missing_full_session"] += 1
                    run.approximate(d, vote, bars["SOXL"], bars)
                mark = bars[run.position.symbol].close if run.position else 0.
                run.mark(d, mark, vote)
            perf, curve = summarize(run, data["SOXL"].close)
            name = f"{key}_{variant}"
            report[name] = perf
            if name == "2026_ytd_hybrid":
                monthly_breakdown(run, curve).to_csv(output / "2026_monthly.csv", index=False)
            year_open = CAPITAL
            for year, rows in curve.groupby([d.year for d in curve.index]):
                equity = rows.equity.iloc[-1]
                annual.append(dict(scenario=name, year=year, initial=year_open, final=equity,
                                   return_pct=100 * (equity / year_open - 1), days=len(rows),
                                   one_minute_days=int((rows.fidelity == "M1").sum()),
                                   five_minute_days=int((rows.fidelity == "M5").sum())))
                year_open = equity
            curve.to_csv(output / f"{name}_equity.csv")
            pd.DataFrame(run.trades).to_csv(output / f"{name}_trades.csv", index=False)
    pd.DataFrame(annual).to_csv(output / "annual_returns.csv", index=False)
    coverage_days = [d for d in days if d >= date(2020, 1, 1)]
    eligible = pd.DataFrame({"date": coverage_days, "minutes": [minute.step.get(d, 0) for d in coverage_days]})
    eligible.to_csv(output / "minute_coverage.csv", index=False)
    report["data"] = {"market_review": False, "night_first": str(min(tradable)), "m1_first": str(first_m1),
                      "night_last": str(max(tradable)),
                      "rescaled": dict(minute.rescaled),
                      "mean_abs_rth_close_error": {m: sum(items) / len(items) if items else None
                                                   for m, items in minute.close_diff.items()}}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


class NoKorea:
    @staticmethod
    def lookup(t):
        return None, "skipped"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.source, args.output)
