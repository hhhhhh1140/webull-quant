"""趋势 / 宏观研判回测用的长周期日线数据。

- 行情：Yahoo Finance 公开日线（SOXL、SOXS、费城半导体指数 ^SOX、^VIX）
- 利率：Yahoo 的 ^IRX（13 周美债收益率，紧跟联邦基金利率，代表加息/降息周期和现金收益）、
  ^FVX（5 年期）、^TNX（10 年期）。FRED 在当前网络下无法访问，所以不用。

缓存在 data/regime/，所有数据只用公开来源，不需要密钥。
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parent.parent / "data" / "regime"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{}"
TICKERS = {"SOX": "^SOX", "SOXL": "SOXL", "SOXS": "SOXS", "VIX": "^VIX",
           "IRX": "^IRX", "FVX": "^FVX", "TNX": "^TNX"}


def _get(url: str, **kw):
    import requests
    for i in range(4):
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60, **kw)
        if r.status_code == 200:
            return r
        time.sleep(2 * (i + 1))
    r.raise_for_status()
    return r


def fetch_yahoo(symbol: str, start: str = "1994-01-01") -> pd.DataFrame:
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(time.time()) + 86400
    j = _get(YAHOO.format(symbol), params={"period1": p1, "period2": p2, "interval": "1d",
                                           "events": "div,split"}).json()["chart"]["result"][0]
    q = j["indicators"]["quote"][0]
    adj = j["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    idx = pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                       "adjclose": adj}, index=idx)
    df.index.name = "date"
    return df.dropna(subset=["close"]).loc[lambda d: ~d.index.duplicated(keep="last")]


def fetch_all() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    for name, sym in TICKERS.items():
        fetch_yahoo(sym).to_csv(DIR / f"{name}.csv")
        print(f"  {name:5s} ok")


def load() -> dict:
    if not (DIR / "SOX.csv").exists():
        fetch_all()
    out = {n: pd.read_csv(DIR / f"{n}.csv", index_col=0, parse_dates=True) for n in TICKERS}
    return out
