"""04:00 ET 仓位研究使用的时点新闻与事件日历。全部方向均以半导体板块为参照。

JSONL 每行记录 headline, source, scope ("semis"), direction (-1/0/1), published_at,
available_at。available_at 为本地首次取得这条新闻及其方向标注的时刻；补录与事后标注
使用实际录入时间。日历 CSV 存 event,published_at,scheduled_at,source，published_at
指可核实的日程公布时刻。两种输入均要求带时区，不依据标题推断方向。
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


_HEADLINE_FIELDS = {"headline", "source", "scope", "direction", "published_at", "available_at"}
_EVENT_FIELDS = {"event", "published_at", "scheduled_at", "source"}


def _text(row, key):
    value = row[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 必须为非空字符串")
    return value.strip()


def _timestamp(row, key):
    value = _text(row, key)
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{key} 必须为带时区的 ISO 8601 时间") from exc
    if ts.tzinfo is None:
        raise ValueError(f"{key} 必须带时区")
    return pd.Timestamp(ts)


@dataclass(frozen=True)
class _Headline:
    published: pd.Timestamp
    available: pd.Timestamp
    source: str
    headline: str
    direction: int


@dataclass(frozen=True)
class _Event:
    published: pd.Timestamp
    scheduled: pd.Timestamp
    source: str
    event: str


def _headlines(path):
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or row.keys() != _HEADLINE_FIELDS:
                    raise ValueError("新闻字段应为 " + ", ".join(sorted(_HEADLINE_FIELDS)))
                if row["scope"] != "semis":
                    raise ValueError("scope 必须为 semis")
                direction = row["direction"]
                if type(direction) is not int or direction not in (-1, 0, 1):
                    raise ValueError("direction 必须为 -1、0 或 1 的整数")
                published, available = _timestamp(row, "published_at"), _timestamp(row, "available_at")
                if available < published:
                    raise ValueError("available_at 早于公开发布时间")
                yield _Headline(published, available, _text(row, "source"), _text(row, "headline"), direction)
            except ValueError as exc:
                raise ValueError(f"{path} 第 {i} 行，{exc}") from exc


def _events(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or sorted(reader.fieldnames) != sorted(_EVENT_FIELDS):
            raise ValueError(f"{path} 日历表头应为 event,published_at,scheduled_at,source")
        for row in reader:
            try:
                if None in row:
                    raise ValueError("日历列数与表头不一致")
                published, scheduled = _timestamp(row, "published_at"), _timestamp(row, "scheduled_at")
                if published > scheduled:
                    raise ValueError("日程公布晚于事件发生时刻")
                yield _Event(published, scheduled, _text(row, "source"), _text(row, "event"))
            except ValueError as exc:
                raise ValueError(f"{path} 第 {reader.line_num} 行，{exc}") from exc


@dataclass(frozen=True)
class NewsDecision:
    coverage: str
    direction: int
    scheduled_risk: bool
    headlines: int

    def confirms(self, side: int) -> bool:
        return self.coverage == "directional" and self.direction == side

    def contradicts(self, side: int) -> bool:
        return self.coverage == "directional" and self.direction == -side

    def requires_cap(self, side: int) -> bool:
        return self.scheduled_risk or self.coverage == "conflict" or self.contradicts(side)


@dataclass(frozen=True)
class NewsBook:
    headlines: tuple[_Headline, ...] = ()
    events: tuple[_Event, ...] = ()

    @classmethod
    def load(cls, path: str | Path | None = None, schedule_path: str | Path | None = None):
        return cls(tuple(_headlines(path)) if path is not None else (),
                   tuple(_events(schedule_path)) if schedule_path is not None else ())

    def for_session(self, cutoff: pd.Timestamp, close_time: pd.Timestamp) -> NewsDecision:
        cutoff, close_time = pd.Timestamp(cutoff), pd.Timestamp(close_time)
        if cutoff.tz is None or close_time.tz is None or close_time < cutoff:
            raise ValueError("决策与收盘时间必须带时区，且收盘不能早于决策")
        start = cutoff - pd.Timedelta(hours=24)
        known = [h.direction for h in self.headlines if start <= h.published and h.available <= cutoff]
        directions = set(known) - {0}
        if len(directions) == 2:
            coverage, direction = "conflict", 0
        elif directions:
            coverage, direction = "directional", directions.pop()
        else:
            coverage, direction = ("neutral" if known else "unknown"), 0
        risk = any(e.published <= cutoff <= e.scheduled <= close_time for e in self.events)
        return NewsDecision(coverage, direction, risk, len(known))
