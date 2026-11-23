"""Pydantic 响应模型。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DeviceInfo(BaseModel):
    type: Literal["npu:0", "cuda:0", "cpu"] = "cpu"
    npu_available: bool = False
    label: str = "CPU"


class StrategyStat(BaseModel):
    name: str
    label: str
    total_cost: float
    violation_rate: float
    ev_served: int
    user_satisfaction: float
    tracking_error: float = 0.0
    transformer_overload: float = 0.0
    is_gnn: bool = False


class SummaryResponse(BaseModel):
    online_evs: int
    total_power_mw: float
    grid_load_mw: float
    transformer_overload_rate: float
    arbitrage_profit: float
    device: DeviceInfo
    demo_mode: bool
    data_source: str
    strategies: list[StrategyStat]
    ts: str


class GraphNode(BaseModel):
    id: str
    kind: Literal["env", "transformer", "cs", "ev"]
    label: str
    meta: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    kind: Literal["env-tr", "tr-cs", "cs-ev"]


class HeteroGraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class TimeseriesResponse(BaseModel):
    steps: list[int]
    price: list[float]
    power: dict[str, list[float]]
    transformer_load: dict[str, list[float]]
    soc_mean: dict[str, list[float]]
    graph: HeteroGraph
    demo_mode: bool
    ts: str


class ActionLog(BaseModel):
    step: int
    cs_id: int
    ev_id: int
    action_kw: float
    reason: str


class ActionsResponse(BaseModel):
    actions: list[ActionLog]
    demo_mode: bool
    ts: str


class DiagnosisResponse(BaseModel):
    summary: str
    risk_points: list[str]
    suggestion: str
    backend: str = "placeholder"
    ts: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    device: DeviceInfo
    demo_mode: bool
    mysql: bool
    redis: bool
    ts: str
