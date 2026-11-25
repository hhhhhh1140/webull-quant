import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from wquant.news_capture import add_arguments, cmd_news_record, record
from wquant.sizing_news import NewsBook


ITEM = dict(source="test fixture", headline="一条测试标题", direction=-1, published_at="2020-01-01T09:00:00+08:00")


def test_first_observation_uses_local_clock_and_retry_is_idempotent(tmp_path):
    path = tmp_path / "headlines.jsonl"
    before = pd.Timestamp.now(tz="UTC")
    created, available = record(path, **ITEM)
    after = pd.Timestamp.now(tz="UTC")
    assert created and before <= pd.Timestamp(available) <= after
    saved = path.read_bytes()
    created, repeated = record(path, **{**ITEM, "published_at": "2020-01-01T01:00:00Z"})
    assert not created and pd.Timestamp(repeated) == pd.Timestamp(available)
    with pytest.raises(ValueError, match="不同的方向"):
        record(path, **{**ITEM, "direction": 1})
    assert path.read_bytes() == saved
    past = pd.Timestamp(ITEM["published_at"]) + pd.Timedelta(hours=1)
    assert NewsBook.load(path).for_session(past, past).coverage == "unknown"


def test_concurrent_calls_and_existing_archive_without_terminal_newline(tmp_path):
    path = tmp_path / "nested" / "headlines.jsonl"
    with ThreadPoolExecutor(max_workers=6) as pool:
        result = list(pool.map(lambda _: record(path, **ITEM)[0], range(10)))
    assert sum(result) == 1
    first = path.read_bytes().rstrip(b"\n")
    path.write_bytes(first)
    record(path, **{**ITEM, "headline": "另一条测试标题\n第二行"})
    assert path.read_bytes().startswith(first + b"\n")
    assert len(NewsBook.load(path).headlines) == 2


def test_invalid_archive_does_not_get_overwritten(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="第 1 行"):
        record(path, **ITEM)
    assert path.read_text(encoding="utf-8") == "{}"


def test_cli_input_has_no_backdate_option_and_checks_timestamps(tmp_path):
    p = argparse.ArgumentParser()
    add_arguments(p.add_subparsers(dest="cmd", required=True))
    path = tmp_path / "news.jsonl"
    args = ["news-record", "--archive", str(path), "--headline", ITEM["headline"], "--source",
            ITEM["source"], "--direction", "-1", "--published-at"]
    with pytest.raises(SystemExit):
        p.parse_args(args + [ITEM["published_at"], "--available-at", ITEM["published_at"]])
    assert cmd_news_record(p.parse_args(args + ["2020-01-01T01:00:00"])) == 1
    assert cmd_news_record(p.parse_args(args + ["2099-01-01T00:00:00Z"])) == 1
    assert not path.exists()
    assert cmd_news_record(p.parse_args(args + [ITEM["published_at"]])) == 0
    assert json.loads(path.read_bytes())["scope"] == "semis"
