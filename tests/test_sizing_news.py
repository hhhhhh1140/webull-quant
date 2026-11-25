import json

import pandas as pd
import pytest

from wquant.sizing_news import NewsBook


CUT = pd.Timestamp("2026-06-17T04:00:00-04:00")
CLOSE = CUT + pd.Timedelta(hours=12)


def headline(direction=1, published=CUT, available=CUT, **changes):
    row = {"headline": "来自测试夹具的半导体新闻", "scope": "semis", "source": "fixture",
           "direction": direction, "published_at": published.isoformat(),
           "available_at": available.isoformat()}
    return {**row, **changes}


def archive(tmp_path, *rows):
    path = tmp_path / "headlines.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_only_actionable_published_news_in_trailing_day_counts(tmp_path):
    one_day, sec = pd.Timedelta(days=1), pd.Timedelta(seconds=1)
    path = archive(tmp_path, headline(published=CUT - one_day), headline(-1, published=CUT - one_day - sec),
                   headline(-1, available=CUT + sec), headline(-1, published=CUT + sec, available=CUT + sec))
    book = NewsBook.load(path)
    known = book.for_session(CUT.tz_convert("UTC"), CLOSE)
    assert (known.coverage, known.headlines, known.direction) == ("directional", 1, 1)
    assert known.confirms(1) and not known.requires_cap(1)
    assert known.contradicts(-1) and known.requires_cap(-1)
    after = book.for_session(CUT + sec, CLOSE)
    assert after.coverage == "directional" and after.direction == -1


def test_unknown_neutral_and_conflicting_coverage_are_distinct(tmp_path):
    unknown = NewsBook.load(None).for_session(CUT, CLOSE)
    assert unknown.coverage == "unknown" and not unknown.confirms(-1)
    neutral = NewsBook.load(archive(tmp_path, headline(0))).for_session(CUT, CLOSE)
    assert neutral.coverage == "neutral" and not neutral.confirms(1)
    mixed = NewsBook.load(archive(tmp_path, headline(), headline(0), headline(-1))).for_session(CUT, CLOSE)
    assert mixed.coverage == "conflict" and mixed.direction == 0 and mixed.headlines == 3
    for side in (-1, 1):
        assert mixed.requires_cap(side) and not mixed.confirms(side)


def test_calendar_uses_published_schedule_and_session_interval(tmp_path):
    schedule = tmp_path / "events.csv"
    schedule.write_text("event,published_at,scheduled_at,source\n"
                        "FOMC,2026-06-17T08:00:00Z,2026-06-17T18:00:00Z,fixture\n", encoding="utf-8")
    book = NewsBook.load(archive(tmp_path, headline()), schedule)
    assert not book.for_session(CUT - pd.Timedelta(seconds=1), CLOSE).scheduled_risk
    assert not book.for_session(CUT, CUT + pd.Timedelta(hours=9)).scheduled_risk
    assert not book.for_session(CUT + pd.Timedelta(hours=11), CLOSE).scheduled_risk
    known = book.for_session(CUT, CLOSE)
    assert known.scheduled_risk and known.confirms(1) and known.requires_cap(1)
    assert NewsBook.load(schedule_path=schedule).for_session(CUT, CLOSE).coverage == "unknown"


@pytest.mark.parametrize("change", [
    {"available_at": "2026-06-17T04:00:00"}, {"published_at": "tomorrow"},
    {"available_at": "2026-06-17T07:59:59Z"}, {"source": ""}, {"headline": " "},
    {"direction": False}, {"direction": "1"}, {"direction": 2}, {"other_field": 2}, {"scope": "tech"},
])
def test_invalid_headlines_fail_at_load(tmp_path, change):
    path = archive(tmp_path, {**headline(), **change})
    with pytest.raises(ValueError, match="第 1 行"):
        NewsBook.load(path)


def test_invalid_schedule_and_cutoff_fail(tmp_path):
    schedule = tmp_path / "events.csv"
    schedule.write_text("event,published_at,scheduled_at,source\n"
                        "FOMC,2026-06-17T17:00:00Z,2026-06-17T16:00:00Z,fixture\n", encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        NewsBook.load(schedule_path=schedule)
    with pytest.raises(ValueError, match="时区"):
        NewsBook().for_session(CUT.tz_localize(None), CLOSE)
