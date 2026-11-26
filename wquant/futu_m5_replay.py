"""Replay the dual-layer strategy on native Futu intraday bars (five minutes by default)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd

from .dual_layer_replay import (CAPITAL, DAY, ET, TICKERS, Bar, Korea, Replay, daily_data,
                                local, monthly_breakdown, native_daily, read_actions, summarize)
from .swing import ensemble_votes

def read_futu(path: Path, bar_minutes: int = 5) -> pd.DataFrame:
    bars = pd.read_csv(path).set_index("time_key").sort_index()
    # Futu US history labels bars by the end of the interval in New York time.
    bars.index = pd.to_datetime(bars.index).tz_localize(ET) - pd.Timedelta(minutes=bar_minutes)
    if bars.index.has_duplicates or not (bars.index.minute % bar_minutes == 0).all():
        raise ValueError(f"重复或未对齐的 {bar_minutes} 分钟 K 线 {path}")
    fields = ["open", "high", "low", "close", "volume"]
    if (bars[fields].isna().any(axis=None) or (bars[fields[:4]] <= 0).any(axis=None)
            or (bars.volume < 0).any() or (bars.high < bars[["open", "close", "low"]].max(axis=1)).any()
            or (bars.low > bars[["open", "close"]].min(axis=1)).any()):
        raise ValueError(f"无效 OHLCV {path}")
    return bars[fields]


def group_bars(prices: Path, session_closes: dict, bar_minutes: int = 5) -> dict:
    dates = {day: {} for day in session_closes}
    # Compute the timestamp boundaries once per trading day.
    boundaries = {day: (local(day, "04:00"), local(day, "09:30"), close,
                        local(day, "17:00" if close.hour == 13 else "20:00"))
                  for day, close in session_closes.items()}
    for symbol in TICKERS:
        bars = read_futu(prices / f"{symbol}_native_M{bar_minutes}.csv.gz", bar_minutes)
        for stamp, o, h, low, c, v in bars.itertuples():
            day = stamp.date() + DAY if stamp.hour >= 20 else stamp.date()
            if day not in dates:
                continue
            pre, regular, close, end = boundaries[day]
            if stamp >= end:
                continue
            label = "OVN" if stamp < pre else "PRE" if stamp < regular else "RTH" if stamp < close else "ATH"
            dates[day].setdefault(stamp, {})[symbol] = Bar(o, h, low, c, v, label)
    dates = {day: dict(sorted(bars.items())) for day, bars in dates.items()}
    # Remove Futu 04:00 ET pre-market open spikes caused by prior-day RTH close reporting.
    ordered_days = sorted(dates)
    for idx in range(1, len(ordered_days)):
        prev_day, day = ordered_days[idx - 1], ordered_days[idx]
        t0355, t0400 = local(day, "03:55"), local(day, "04:00")
        t_prev_rth = max((t for t, b in dates[prev_day].items() if any(v.session == "RTH" for v in b.values())), default=None)
        if t0400 not in dates[day] or t0355 not in dates[day] or t_prev_rth is None:
            continue
        for sym in TICKERS:
            if sym not in dates[day][t0400] or sym not in dates[day][t0355] or sym not in dates[prev_day][t_prev_rth]:
                continue
            b = dates[day][t0400][sym]
            c355 = dates[day][t0355][sym].close
            prth = dates[prev_day][t_prev_rth][sym].close
            if prth > 0 and c355 > 0 and abs(b.open / prth - 1) < .005 and abs(b.open / c355 - 1) > .015:
                new_o = c355
                new_l = min(new_o, b.close) if abs(b.low / b.open - 1) < .002 else b.low
                new_h = max(new_o, b.close) if abs(b.high / b.open - 1) < .002 else b.high
                dates[day][t0400][sym] = Bar(new_o, max(new_h, new_o, b.close), min(new_l, new_o, b.close),
                                             b.close, b.volume, b.session)
    return dates


def replay_runs(prices: Path, reference: Path, start: date, end: date, runs: dict[str, Replay], *,
                bar_minutes: int = 5, one_minute_overlay: tuple[Path, set[date]] | None = None):
    calendar = xcals.get_calendar("XNYS", start=start - DAY * 14, end=end + DAY * 14)
    date_range = calendar.sessions_in_range(start, end)
    if date_range.empty:
        raise ValueError("回测区间没有美股交易日")
    previous = calendar.previous_session(date_range[0])
    closes = {session.date(): calendar.session_close(session).tz_convert(ET) for session in [previous, *date_range]}
    book = group_bars(prices, closes, bar_minutes)
    fine_book, fine_days = {}, set()
    if one_minute_overlay is not None:
        fine_source, fine_days = one_minute_overlay
        fine_book = group_bars(fine_source, closes, 1)
    korea = Korea(reference)
    votes = ensemble_votes(daily_data(reference, "SOXX"))
    splits = {s: read_actions(reference, s, "splits") for s in TICKERS}
    distributions = {s: read_actions(reference, s, "dividends") for s in TICKERS}
    for sym, events in distributions.items():
        for day, amount in events.items():
            for ex, ratio in splits[sym].items():
                if day < ex:
                    amount *= ratio
            events[day] = amount
    controls = {s: native_daily(daily_data(reference, s), splits[s]).close for s in TICKERS}

    coverage, benchmark, close_errors = [], {}, {}
    last_rth_close: dict[str, float] = {}
    for day, regular_close in closes.items():
        day_book, step = (fine_book, 1) if day in fine_days else (book, bar_minutes)
        bars = day_book[day]
        rth_grid = pd.date_range(local(day, "09:30"), regular_close, freq=f"{step}min", inclusive="left")
        session_counts = Counter((symbol, bar.session) for symbols in bars.values() for symbol, bar in symbols.items())
        rth_open, rth_close = {}, {}
        for sym in TICKERS:
            rth = [bars[t][sym] for t in rth_grid if sym in bars.get(t, {})]
            if (len(rth) < len(rth_grid) * .9 or sym not in bars.get(rth_grid[0], {})
                    or sym not in bars.get(rth_grid[-1], {})):
                raise ValueError(f"{day} {sym} 常规盘 M{step} 覆盖不足 {len(rth)}/{len(rth_grid)}")
            rth_open[sym], rth_close[sym] = rth[0].open, rth[-1].close
            if day in controls[sym].index:
                error = abs(rth_close[sym] / controls[sym].at[day] - 1)
                if error > .015:
                    close_errors[f"{day} {sym}"] = round(error, 5)
        if day == previous.date():
            last_rth_close = rth_close
            continue

        prev_day = calendar.previous_session(pd.Timestamp(day)).date()
        if prev_day not in votes.index or pd.isna(votes.at[prev_day]):
            raise ValueError(f"缺少 {prev_day} 的 SOXX 日线趋势票")
        split = splits["SOXL"].get(day, 1)
        prep = [row["SOXL"] for row in day_book[prev_day].values() if "SOXL" in row and row["SOXL"].session == "ATH"]
        if split != 1:
            prep = [Bar(b.open / split, b.high / split, b.low / split, b.close / split, b.volume * split, b.session)
                    for b in prep]
        vote = votes.at[prev_day]
        for run in runs.values():
            run.reset_day(day, step, splits, distributions, rth_open)
            if vote < 2:
                run.core_stop = None
            last = run.replay_day(day, prep, vote, step, bars, last_rth_close["SOXL"] / split, korea)
            value = last.get(run.position.symbol, rth_close[run.position.symbol]) if run.position else 0.0
            run.mark(day, value, vote)
        last_rth_close = rth_close
        benchmark[day] = rth_close["SOXL"]
        for ex, ratio in splits["SOXL"].items():
            if day < ex <= end:
                benchmark[day] /= ratio
        coverage.append({"date": day, "rth_expected": len(rth_grid),
                         **{f"{symbol}_{session}": session_counts[symbol, session]
                            for symbol in TICKERS for session in ("OVN", "PRE", "RTH", "ATH")}})
    if close_errors:
        raise ValueError(f"M{bar_minutes} 收盘价与未复权日线差异超过 1.5% {len(close_errors)} 项，样例 {list(close_errors.items())[:10]}")
    return pd.Series(benchmark), pd.DataFrame(coverage)


def write_report(run: Replay, benchmark: pd.Series, coverage: pd.DataFrame, output: Path) -> dict:
    metrics, curve = summarize(run, benchmark)
    annual_rows, opening = [], CAPITAL
    for year, equity in curve.equity.groupby(lambda day: day.year):
        closing = equity.iloc[-1]
        annual_rows.append({"year": year, "start": equity.index[0], "end": equity.index[-1], "days": len(equity),
                            "opening_equity": opening, "closing_equity": closing, "return": closing / opening - 1})
        opening = closing
    equity = pd.concat([pd.Series({"initial": CAPITAL}), curve.equity])
    trough = (equity / equity.cummax() - 1).idxmin()
    peak = equity.loc[:trough].idxmax()
    metrics["drawdown_period"] = {"peak": str(peak), "trough": str(trough),
                                  "peak_equity": float(equity.at[peak]), "trough_equity": float(equity.at[trough])}
    steps = [f"K_{step}M" for step in (1, 5) if run.stats[f"days_M{step}"]]
    metrics["data"] = {"source": f"Futu {'+'.join(steps)} / ALL / NONE", "daily_price_fallback_days": 0,
                       "sessions_with_both_overnight": int(((coverage.SOXL_OVN > 0) & (coverage.SOXS_OVN > 0)).sum()),
                       "sessions_with_partial_rth_grid": int(((coverage.SOXL_RTH < coverage.rth_expected)
                                                               | (coverage.SOXS_RTH < coverage.rth_expected)).sum()),
                       "order_limit_offset_bps": round(run.offset * 10_000),
                       "scan_end_et": run.scan_end, "allow_second_entry": run.allow_reentry,
                       "uses_rate_gate": run.short_gate is not None}
    output.mkdir(parents=True, exist_ok=True)
    curve.to_csv(output / "equity.csv")
    pd.DataFrame(run.trades).to_csv(output / "trades.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(output / "annual_returns.csv", index=False)
    monthly_breakdown(run, curve).to_csv(output / "monthly_returns.csv", index=False)
    coverage.to_csv(output / "coverage.csv", index=False)
    (output / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_replay(prices: Path, reference: Path, output: Path, start: date, end: date, *, bar_minutes: int = 5) -> dict:
    run = Replay(.003)
    benchmark, coverage = replay_runs(prices, reference, start, end, {"baseline": run}, bar_minutes=bar_minutes)
    return write_report(run, benchmark, coverage, output)


def compare_paper_profiles(prices: Path, reference: Path, rates: Path, output: Path,
                           start: date, end: date, *, bar_minutes: int = 5,
                           one_minute_overlay: tuple[Path, set[date]] | None = None) -> dict:
    from .config import load_settings
    from .paper_arena import PROFILES, load_profile_settings
    from .swing_live import eval_macro_short_gate, swing_cfg

    base = load_settings(environment="uat")
    date_range = pd.date_range(start, end, freq="D").date
    earliest_rate = start - DAY * 400
    series: dict[str, pd.Series] = {}
    gates: dict[tuple, dict] = {}
    runs = {}
    for profile in PROFILES:
        if not profile.config_file.is_file():
            raise FileNotFoundError(profile.config_file)
        cfg = swing_cfg(load_profile_settings(base, profile))
        tactical = cfg["tactical"]
        trend = tactical["trend_switch"]
        macro = tactical["macro_short_gate"]
        gate = None
        if macro.get("enabled"):
            key = (tuple(macro["symbols"]), int(macro["sma_window"]))
            for symbol in key[0]:
                if symbol not in series:
                    rows = pd.read_csv(rates / f"{symbol.lstrip('^')}.csv", usecols=["date", "close"])
                    index = pd.to_datetime(rows.date).dt.date
                    values = pd.Series(rows.close.to_numpy(), index=index).dropna().sort_index()
                    series[symbol] = values.loc[earliest_rate:end]
                    if len(series[symbol].loc[:start - DAY]) < key[1]:
                        raise ValueError(f"{symbol} 利率历史缺少均线预热数据")
            if key not in gates:
                gates[key] = {d: eval_macro_short_gate(d, macro, fetcher=series.__getitem__) for d in date_range}
            gate = {d: allowed for d, (allowed, _) in gates[key].items()}
        runs[profile.key] = Replay(cfg["order"]["limit_offset_bps"] / 10_000, trend["trail_pct"],
                                   reentry_position_frac=tactical["reentry_position_frac"],
                                   reentry_trail_pct=tactical["reentry_trail_pct"],
                                   first_trail_pct=tactical["trail_pct"], reentry_stop_pct=tactical["reentry_stop_pct"],
                                   core_rebuy_session=trend["rebuy_session"], short_gate=gate,
                                   scan_end=tactical["path"]["scan_to"],
                                   allow_reentry=tactical["reentry"] and tactical["max_reentries"] != 0)

    benchmark, coverage = replay_runs(prices, reference, start, end, runs, bar_minutes=bar_minutes,
                                      one_minute_overlay=one_minute_overlay)
    result = {profile.key: write_report(runs[profile.key], benchmark, coverage, output / profile.key)
              for profile in PROFILES}
    for profile in PROFILES:
        result[profile.key]["label"] = profile.label
    gate_rows = [{"date": day, "symbols": ",".join(symbols), "sma_window": window,
                  "allowed": lookup[day][0], "detail": lookup[day][1],
                  **{f"{sym}_date": str(series[sym].loc[:day - DAY].index[-1]) for sym in symbols}}
                 for (symbols, window), lookup in gates.items() for day in coverage.date]
    if gate_rows:
        pd.DataFrame(gate_rows).to_csv(output / "rate_gate.csv", index=False)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prices", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--bar-minutes", type=int, choices=(1, 5), default=5)
    parser.add_argument("--paper-profiles", action="store_true")
    parser.add_argument("--rates", type=Path, help="IRX/TNX 日线目录，用于 --paper-profiles")
    args = parser.parse_args()
    if args.paper_profiles and args.rates is None:
        parser.error("--paper-profiles 需要 --rates")
    report = (compare_paper_profiles(args.prices, args.reference, args.rates, args.output, args.start, args.end,
                                    bar_minutes=args.bar_minutes)
              if args.paper_profiles else run_replay(args.prices, args.reference, args.output, args.start, args.end,
                                                      bar_minutes=args.bar_minutes))
    print(json.dumps(report, ensure_ascii=False, indent=2))
