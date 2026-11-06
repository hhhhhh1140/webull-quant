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
    def groups(self) -> dict[str, str]:
        """{symbol: 板块}。universe 可以写成列表（全部归入 default）或按板块分组的字典。"""
        u = self.raw.get("universe", []) or []
        if isinstance(u, dict):
            return {s.upper(): g for g, syms in u.items() for s in (syms or [])}
        return {s.upper(): "default" for s in u}

    @property
    def universe(self) -> list[str]:
        return list(self.groups)

    def instrument(self, symbol: str) -> dict:
        """{leverage, inverse}，没配置的按普通股票处理。"""
        meta = (self.raw.get("instruments") or {}).get(symbol.upper()) or {}
        return {"leverage": float(meta.get("leverage", 1)), "inverse": bool(meta.get("inverse", False))}

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
    # 沙盒和实盘用不同的密钥：uat 读 WEBULL_SANDBOX_APP_KEY/SECRET，prod 读 WEBULL_APP_KEY/SECRET
    prefix = "WEBULL_SANDBOX_" if env == "uat" else "WEBULL_"
    key = os.environ.get(f"{prefix}APP_KEY", "").strip()
    secret = os.environ.get(f"{prefix}APP_SECRET", "").strip()
    # 每套密钥单独一个 token 目录，SDK 默认共用一个文件会互相覆盖
    base = Path(os.environ.get("WEBULL_TOKEN_DIR", str(ROOT / "conf")))
    return Settings(
        raw=raw,
        app_key=key,
        app_secret=secret,
        region_id=os.environ.get("WEBULL_REGION_ID", "us").strip().lower(),
        environment=env,
        token_dir=str(base / f"{env}-{key[:8]}") if key else str(base),
    )
