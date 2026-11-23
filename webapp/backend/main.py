"""智充图驭 Web 驾驶舱 FastAPI 入口。

路由：
  GET  /api/health
  GET  /api/dashboard/summary
  GET  /api/dashboard/timeseries
  GET  /api/dashboard/actions
  POST /api/dashboard/diagnosis
  GET  /                      (静态托管 frontend/index.html)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import cache, db
from .config import CACHE_TTL_SUMMARY, CACHE_TTL_TIMESERIES
from .device_probe import probe_device
from .schemas import (
    HealthResponse, SummaryResponse, TimeseriesResponse, ActionsResponse,
    DiagnosisResponse,
)
from .data_loader import load_real_data
from .demo_data import build_demo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
_logger = logging.getLogger("webapp")

app = FastAPI(
    title="智充图驭 Web 驾驶舱",
    description="大规模电动汽车智能充电调度云平台可视化大盘",
    version="1.0.0",
)

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_bundle():
    """优先真实数据，回退演示数据。返回 (summary, timeseries, actions, demo_mode)。"""
    device = probe_device()
    real = load_real_data(device)
    if real is not None:
        _logger.info("data source: real (demo_mode=false)")
        return real.summary, real.timeseries, real.actions, False
    demo = build_demo(device)
    _logger.info("data source: demo (demo_mode=true)")
    return demo.summary, demo.timeseries, demo.actions, True


@app.get("/api/health", response_model=HealthResponse)
async def health():
    device = probe_device()
    return HealthResponse(
        status="ok",
        device=device,
        demo_mode=not _has_real_data(),
        mysql=db.status()["mysql_enabled"],
        redis=cache.status()["redis_enabled"],
        ts=_now(),
    )


def _has_real_data() -> bool:
    device = probe_device()
    return load_real_data(device) is not None


@app.get("/api/dashboard/summary", response_model=SummaryResponse)
async def dashboard_summary():
    cached = cache.get("summary")
    if cached:
        return JSONResponse(content=json.loads(cached))
    summary, _, _, _ = _load_bundle()
    payload = summary.model_dump()
    cache.setex("summary", CACHE_TTL_SUMMARY, json.dumps(payload, ensure_ascii=False))
    return summary


@app.get("/api/dashboard/timeseries", response_model=TimeseriesResponse)
async def dashboard_timeseries():
    cached = cache.get("timeseries")
    if cached:
        return JSONResponse(content=json.loads(cached))
    _, timeseries, _, _ = _load_bundle()
    payload = timeseries.model_dump()
    cache.setex("timeseries", CACHE_TTL_TIMESERIES,
                json.dumps(payload, ensure_ascii=False))
    return timeseries


@app.get("/api/dashboard/actions", response_model=ActionsResponse)
async def dashboard_actions():
    cached = cache.get("actions")
    if cached:
        return JSONResponse(content=json.loads(cached))
    _, _, actions, _ = _load_bundle()
    payload = actions.model_dump()
    cache.setex("actions", CACHE_TTL_SUMMARY,
                json.dumps(payload, ensure_ascii=False))
    return actions


@app.post("/api/dashboard/diagnosis", response_model=DiagnosisResponse)
async def dashboard_diagnosis(req: Request):
    """AI 调度诊断摘要预留接口。当前返回占位摘要，待接入华为云 MaaS。"""
    try:
        body = await req.json()
    except Exception:
        body = {}
    user_text = str(body.get("text", ""))[:500]
    return DiagnosisResponse(
        summary=f"已收到诊断请求（{len(user_text)} 字）。当前为占位实现，待接入华为云 MaaS 大模型推理服务。",
        risk_points=[
            "配变越限风险（待 MaaS 分析实时负载曲线）",
            "用户充电满意度与电费套利权衡（待 MaaS 多目标推理）",
        ],
        suggestion="建议接入华为云 MaaS 后，将本接口改为转发至推理端点，输入 112 步功率/电价/配变负载序列，输出结构化调度建议。",
        backend="placeholder",
        ts=_now(),
    )


if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    idx = _FRONTEND_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return HTMLResponse("<h1>frontend/index.html missing</h1>", status_code=503)


@app.on_event("startup")
async def _startup():
    device = probe_device()
    _logger.info("device: %s (%s)", device.type, device.label)
    _logger.info("db: %s", db.status())
    _logger.info("cache: %s", cache.status())
    _logger.info("frontend dir: %s (exists=%s)", _FRONTEND_DIR, _FRONTEND_DIR.exists())
