from datetime import date

import numpy as np
import pandas as pd

from wquant.bear_rally import Morning, open_replay, rally_orders, replay
from wquant.scalp_data import ET


def morning():
    idx = pd.date_range("2026-11-01 20:00", "2026-11-02 10:30", freq="5min", tz=ET)
    labels = np.where(idx < pd.Timestamp("2026-11-02 04:00", tz=ET), "OVN",
                      np.where(idx < pd.Timestamp("2026-11-02 09:30", tz=ET), "PRE", "RTH"))
    bars = pd.DataFrame({"open": 100., "high": 100., "low": 100., "close": 100.,
                         "volume": 1000, "session": labels}, index=idx)
    return Morning(date(2026,11,2), -.05, bars, bars.copy(), True)


def test_night_rally_needs_opposite_moves_and_uses_previous_completed_bar():
    d = morning()
    cols = ["open","high","low","close"]
    d.S.loc[d.S.index[11:13],cols] = 99.
    assert rally_orders(d) == []
    d.L.loc[d.L.index[11:],cols] = 102.
    e, = rally_orders(d)
    assert e.bar == 12 and e.side == -1 and 99 < e.max_buy_price < 100
    assert rally_orders(d, delay=1)[0].bar == 13
    assert len(replay([d], 30)[1]) == 1
    assert replay([d], 30, delay=1)[1] == []  # 后一根跳过信号时的最高买价


def test_night_stop_and_matched_open_entry_are_separate_positions():
    d = morning()
    cols = ["open","high","low","close"]
    d.L.loc[d.L.index[11:],cols] = 102.
    d.S.loc[d.S.index[11:13],cols] = 99.
    d.S.loc[d.S.index[20],cols] = 90.
    d.S.loc[d.S.index[-7],cols] = 104.  # 10:00 ET
    _, trades = replay([d],30)
    assert trades[0].exit_time == d.S.index[20] and trades[0].pnl < 0
    m = open_replay([d])
    assert m["n"] == 1 and m["net"] > 0
