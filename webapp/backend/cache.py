"""Redis(华为云 DCS) 缓存封装；未配置时降级为进程内 LRU dict。"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from .config import REDIS_URL, redis_enabled

_logger = logging.getLogger("webapp.cache")
_local_lock = threading.Lock()
_local_store: OrderedDict[str, tuple[float, str]] = OrderedDict()
_LOCAL_MAX = 256

try:
    import redis  # type: ignore
    _REDIS_LIB = True
except Exception:
    _REDIS_LIB = False

_redis_client = None


def _client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not _REDIS_LIB:
        raise RuntimeError("redis 库未安装")
    _redis_client = redis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
    return _redis_client


def get(key: str) -> str | None:
    try:
        if redis_enabled():
            return _client().get(key)
    except Exception as exc:
        _logger.warning("redis get failed, fallback local: %s", exc)
    with _local_lock:
        if key in _local_store:
            exp, val = _local_store.pop(key)
            if exp > time.time():
                _local_store[key] = (exp, val)
                return val
        return None


def setex(key: str, ttl: int, value: str) -> None:
    try:
        if redis_enabled():
            _client().setex(key, ttl, value)
            return
    except Exception as exc:
        _logger.warning("redis setex failed, fallback local: %s", exc)
    with _local_lock:
        _local_store[key] = (time.time() + ttl, value)
        _local_store.move_to_end(key)
        while len(_local_store) > _LOCAL_MAX:
            _local_store.popitem(last=False)


def status() -> dict[str, Any]:
    return {
        "redis_enabled": redis_enabled(),
        "redis_lib": _REDIS_LIB,
        "fallback": "local-lru" if not redis_enabled() else "redis",
    }
