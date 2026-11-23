"""MySQL(华为云 RDS) 连接封装；未配置时降级为本地 SQLite。

仅用标准库 sqlite3 + 可选 pymysql，避免硬依赖。webapp 自身不强制安装 pymysql。
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Any

from .config import MYSQL_CONFIG, LOCAL_DB_PATH, mysql_enabled

_logger = logging.getLogger("webapp.db")
_local_lock = threading.Lock()
_local_conn: sqlite3.Connection | None = None

try:
    import pymysql  # type: ignore
    _PYMYSQL = True
except Exception:
    _PYMYSQL = False


def _sqlite_conn() -> sqlite3.Connection:
    global _local_conn
    if _local_conn is None:
        _local_conn = sqlite3.connect(str(LOCAL_DB_PATH), check_same_thread=False)
        _local_conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, ts TEXT)"
        )
        _local_conn.commit()
        _logger.info("sqlite fallback at %s", LOCAL_DB_PATH)
    return _local_conn


def _mysql_conn():
    if not _PYMYSQL:
        raise RuntimeError("pymysql 未安装，请 pip install pymysql 或配置 MYSQL_*")
    return pymysql.connect(
        host=MYSQL_CONFIG["host"], port=MYSQL_CONFIG["port"],
        user=MYSQL_CONFIG["user"], password=MYSQL_CONFIG["password"],
        database=MYSQL_CONFIG["db"], charset="utf8mb4",
        autocommit=True,
    )


def put(key: str, value: str, ts: str) -> None:
    try:
        if mysql_enabled():
            conn = _mysql_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "REPLACE INTO kv(k, v, ts) VALUES(%s,%s,%s)", (key, value, ts))
            conn.close()
            return
    except Exception as exc:
        _logger.warning("mysql put failed, fallback sqlite: %s", exc)
    with _local_lock:
        c = _sqlite_conn()
        c.execute("REPLACE INTO kv(k, v, ts) VALUES(?,?,?)", (key, value, ts))
        c.commit()


def get(key: str) -> str | None:
    try:
        if mysql_enabled():
            conn = _mysql_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT v FROM kv WHERE k=%s", (key,))
                row = cur.fetchone()
            conn.close()
            return row[0] if row else None
    except Exception as exc:
        _logger.warning("mysql get failed, fallback sqlite: %s", exc)
    with _local_lock:
        c = _sqlite_conn()
        row = c.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return row[0] if row else None


def status() -> dict[str, Any]:
    return {
        "mysql_enabled": mysql_enabled(),
        "mysql_driver": _PYMYSQL,
        "fallback": "sqlite" if not mysql_enabled() else "mysql",
        "local_db": str(LOCAL_DB_PATH),
    }
