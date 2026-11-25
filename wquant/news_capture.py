"""在本机首次录入带方向的半导体标题时保存可用时间，供后续仓位回放。"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT
from .sizing_news import NewsBook


DEFAULT_ARCHIVE = ROOT / "data" / "news" / "semis.jsonl"


def record(path: str | Path, *, published_at: str, source: str, headline: str,
           direction: int) -> tuple[bool, str]:
    """返回 (是否首次写入, 首次可用时间)。相同标题和来源的不同发布时间视作不同稿件。"""
    import fcntl
    source, headline = source.strip(), headline.strip()
    if not source or not headline:
        raise ValueError("必须填写可核实的新闻来源及原始标题")
    if type(direction) is not int or direction not in (-1, 0, 1):
        raise ValueError("半导体板块方向必须为 -1、0 或 1")
    try:
        published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("发布时间必须为带时区的 ISO 8601 时间") from exc
    if published.tzinfo is None:
        raise ValueError("发布时间必须带时区")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        now = datetime.now(timezone.utc)
        if published > now:
            raise ValueError("发布时间晚于本机当前时间")
        existing = NewsBook.load(path) if path.exists() else NewsBook()
        seen = [h for h in existing.headlines if h.published == published
                and h.source == source and h.headline == headline]
        if seen:
            if any(h.direction != direction for h in seen):
                raise ValueError("这条新闻已有不同的方向记录，原始归档已保留")
            return False, min(h.available for h in seen).isoformat()

        row = {"headline": headline, "scope": "semis", "source": source, "direction": direction,
               "published_at": published.astimezone(timezone.utc).isoformat(), "available_at": now.isoformat()}
        # 用相邻临时文件替换，读者总能看到完整 JSONL；保留既有记录的字节和可用时间。
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            tmp = Path(temp.name)
            try:
                if path.exists():
                    with path.open("rb") as old:
                        shutil.copyfileobj(old, temp)
                        if old.tell():
                            old.seek(-1, os.SEEK_END)
                            if old.read(1) != b"\n":
                                temp.write(b"\n")
                temp.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
                temp.flush()
                os.fsync(temp.fileno())
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        return True, now.isoformat()


def add_arguments(sub):
    p = sub.add_parser("news-record", help="归档一条已公开的半导体板块新闻与当下方向标注")
    p.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="JSONL 文件，默认 data/news/semis.jsonl")
    p.add_argument("--published-at", required=True, help="来源标明的发布时间（ISO 8601，含时区）")
    p.add_argument("--source", required=True, help="可核实的原文链接或来源")
    p.add_argument("--headline", required=True, help="原文标题")
    p.add_argument("--direction", required=True, type=int, choices=(-1, 0, 1), help="对半导体板块的判断，-1 偏空、0 中性、1 偏多")


def cmd_news_record(args, settings=None):
    try:
        created, available = record(args.archive, published_at=args.published_at, headline=args.headline,
                                    source=args.source, direction=args.direction)
    except (OSError, ValueError) as exc:
        print(f"新闻归档失败，{exc}", file=sys.stderr)
        return 1
    print(f"{'已保存新闻' if created else '已存在相同新闻'}，首次可用时间 {available}，归档 {args.archive}")
    return 0
