"""配置加载：config.yaml（策略/风控）+ .env（密钥/环境）。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    raw: dict[str, Any]
    app_key: str = ""
    app_secret: str = ""
    region_id: str = "us"
    environment: str = "uat"
    token_dir: str = "./conf"

    @property
    def universe(self) -> list[str]:
        return [s.upper() for s in self.raw.get("universe", [])]

    @property
    def mode(self) -> str:
        return self.raw.get("mode", "dry_run")

    @property
    def live_trading_enabled(self) -> bool:
        return bool(self.raw.get("live_trading_enabled", False))

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}) or {})

    @property
    def has_webull_credentials(self) -> bool:
        return bool(self.app_key and self.app_secret)


def load_settings(config_path: str | os.PathLike | None = None) -> Settings:
    load_dotenv(ROOT / ".env")
    path = Path(config_path) if config_path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    env = os.environ.get("WEBULL_ENVIRONMENT", "uat").strip().lower()
    if env not in ("uat", "prod"):
        raise ValueError(f"WEBULL_ENVIRONMENT 必须是 uat 或 prod，当前为 {env!r}")
    return Settings(
        raw=raw,
        app_key=os.environ.get("WEBULL_APP_KEY", "").strip(),
        app_secret=os.environ.get("WEBULL_APP_SECRET", "").strip(),
        region_id=os.environ.get("WEBULL_REGION_ID", "us").strip().lower(),
        environment=env,
        token_dir=os.environ.get("WEBULL_TOKEN_DIR", str(ROOT / "conf")),
    )
