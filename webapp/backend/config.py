"""集中读取环境变量与路径配置。

- 仓库根目录自动推断为 webapp 上级目录（可被 EVGNN_REPO_ROOT 覆盖）。
- 云服务（MySQL/Redis）通过环境变量开关；未配置时 db/cache 模块优雅降级。
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
_WEBAPP_ROOT = _HERE.parent
load_dotenv(_WEBAPP_ROOT / ".env")


def _infer_repo_root() -> Path:
    explicit = os.environ.get("EVGNN_REPO_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return _WEBAPP_ROOT.parent.resolve()


REPO_ROOT: Path = _infer_repo_root()
SAVED_MODELS_DIR: Path = REPO_ROOT / "saved_models"
RESULTS_DIR: Path = REPO_ROOT / "results"
CONFIG_FILES_DIR: Path = REPO_ROOT / "config_files"
LOCAL_DB_PATH: Path = _WEBAPP_ROOT / "local.db"
EVAL_SNAPSHOT_PATH: Path = _WEBAPP_ROOT / "data" / "eval_snapshot.json"

SIMULATION_LENGTH = 112
DEFAULT_CS_COUNT = 25
DEFAULT_TRANSFORMERS = 3
TIMESCALE_MIN = 15

MYSQL_CONFIG = {
    "host": os.environ.get("MYSQL_HOST"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER"),
    "password": os.environ.get("MYSQL_PASSWORD"),
    "db": os.environ.get("MYSQL_DB", "evgnn"),
}

REDIS_URL: str | None = os.environ.get("REDIS_URL")

DEVICE_FORCE: str | None = os.environ.get("WEBAPP_DEVICE_FORCE")

CACHE_TTL_SUMMARY = int(os.environ.get("WEBAPP_CACHE_TTL_SUMMARY", "30"))
CACHE_TTL_TIMESERIES = int(os.environ.get("WEBAPP_CACHE_TTL_TIMESERIES", "60"))


def mysql_enabled() -> bool:
    return bool(MYSQL_CONFIG["host"] and MYSQL_CONFIG["user"])


def redis_enabled() -> bool:
    return bool(REDIS_URL)
