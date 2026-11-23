"""与 EV2Gym 112 步仿真对齐的结构化演示数据生成器。

参数对齐 config_files/PublicPST.yaml:
  simulation_length=112, timescale=15min, 25 CS, 3 transformers, start hour=5.

所有曲线均为合成数据，但量纲与 EV2Gym stats 字段一致：
  - power: 单桩充电功率 kW（V2G 关闭，PublicPST 场景，单桩 11-22 kW）
  - price: 分时电价 元/kWh（峰平谷三段）
  - transformer_load: 配变负载率 0~1.2（>1 越限）
  - soc: EV 电池荷电状态 0~1
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import (
    SIMULATION_LENGTH, DEFAULT_CS_COUNT, DEFAULT_TRANSFORMERS, TIMESCALE_MIN,
)
from .schemas import (
    SummaryResponse, TimeseriesResponse, ActionsResponse, ActionLog,
    StrategyStat, HeteroGraph, GraphNode, GraphEdge, DeviceInfo,
)

STRATEGIES = ("unordered", "rule", "gnn")
STRATEGY_LABELS = {
    "unordered": "无序充电",
    "rule": "规则控制",
    "gnn": "EV-GNN(Ascend NPU)",
}


def _time_axis() -> np.ndarray:
    """每步的绝对小时（从 5:00 起，步长 15min，共 112 步 → 28h 窗口）。"""
    start_hour = 5.0
    step_hour = TIMESCALE_MIN / 60.0
    return start_hour + np.arange(SIMULATION_LENGTH) * step_hour


def _price_curve(hours: np.ndarray) -> np.ndarray:
    """分时电价：谷 0:00-8:00 = 0.35，平 8:00-12/18:00 = 0.78，峰 12-18/19-23 = 1.25。"""
    h = hours % 24.0
    price = np.where(
        (h < 8.0), 0.35,
        np.where((h >= 8.0) & (h < 12.0), 0.78,
                 np.where((h >= 12.0) & (h < 18.0), 1.25,
                          np.where((h >= 18.0) & (h < 19.0), 0.78,
                                   np.where((h >= 19.0) & (h < 23.0), 1.25, 0.35)))))
    return price


def _ev_arrival_profile(hours: np.ndarray) -> np.ndarray:
    """在网 EV 数量曲线：早高峰 7-9 与晚高峰 16-19 双峰，归一化到 [0,1]。"""
    h = hours % 24.0
    morning = np.exp(-((h - 8.0) / 1.5) ** 2)
    evening = np.exp(-((h - 17.5) / 1.8) ** 2)
    profile = 0.55 * morning + 0.75 * evening + 0.08
    return profile / profile.max()


def _power_curve(hours: np.ndarray, strategy: str) -> np.ndarray:
    """单策略 112 步总功率 MW。"""
    arrival = _ev_arrival_profile(hours)
    price = _price_curve(hours)
    base = 0.0
    if strategy == "unordered":
        # 无序：到达即满功率充，叠加尖峰
        base = arrival * 0.95 + 0.05
    elif strategy == "rule":
        # 规则：谷时多充、峰时少充，平滑
        prefer = np.where(price < 0.5, 1.0, np.where(price < 1.0, 0.6, 0.2))
        base = arrival * prefer * 0.85 + 0.05
    elif strategy == "gnn":
        # GNN：进一步削峰填谷，跟踪电价最优
        prefer = np.where(price < 0.5, 1.0, np.where(price < 1.0, 0.45, 0.08))
        base = arrival * prefer * 0.78 + 0.04
    # 25 CS × 22 kW 峰值 ≈ 0.55 MW；映射到 MW
    return base * DEFAULT_CS_COUNT * 0.022


def _transformer_load(hours: np.ndarray, strategy: str) -> np.ndarray:
    """3 配变负载率，>1 越限。"""
    power_mw = _power_curve(hours, strategy)
    # 假设 3 配变总容量 0.6 MW
    return power_mw / 0.6 / DEFAULT_TRANSFORMERS * DEFAULT_TRANSFORMERS * 0.9 + 0.05


def _soc_curve(hours: np.ndarray, strategy: str) -> np.ndarray:
    """平均 SoC 0~1。"""
    arrival = _ev_arrival_profile(hours)
    if strategy == "unordered":
        base = 0.3 + 0.5 * np.cumsum(arrival) / SIMULATION_LENGTH
    elif strategy == "rule":
        base = 0.35 + 0.55 * np.cumsum(arrival * (1 + 0.2 * np.sin(hours))) / SIMULATION_LENGTH
    else:
        base = 0.4 + 0.6 * np.cumsum(arrival * (1 + 0.3 * np.sin(hours * 0.5))) / SIMULATION_LENGTH
    return np.clip(base, 0.05, 0.98)


def _strategy_stats(strategy: str) -> StrategyStat:
    hours = _time_axis()
    power = _power_curve(hours, strategy)
    tload = _transformer_load(hours, strategy)
    arrival = _ev_arrival_profile(hours)
    price = _price_curve(hours)
    energy_kwh = float(np.sum(power) * TIMESCALE_MIN / 60.0 * 1000.0)
    cost = float(np.sum(power * 1000.0 * price) * TIMESCALE_MIN / 60.0)
    violation = float(np.mean(tload > 1.0))
    overload = float(np.sum(np.clip(tload - 1.0, 0, None)))
    served = int(round(float(np.mean(arrival)) * DEFAULT_CS_COUNT * 4.2))
    if strategy == "unordered":
        sat, te = 0.72, 0.31
    elif strategy == "rule":
        sat, te = 0.86, 0.18
    else:
        sat, te = 0.93, 0.07
    return StrategyStat(
        name=strategy, label=STRATEGY_LABELS[strategy],
        total_cost=round(cost, 2),
        violation_rate=round(violation, 4),
        ev_served=served,
        user_satisfaction=round(sat, 3),
        tracking_error=round(te, 3),
        transformer_overload=round(overload, 3),
        is_gnn=(strategy == "gnn"),
    )


def _hetero_graph() -> HeteroGraph:
    nodes: list[GraphNode] = [GraphNode(id="env", kind="env", label="环境节点",
                                        meta={"sim_length": SIMULATION_LENGTH})]
    for t in range(DEFAULT_TRANSFORMERS):
        nodes.append(GraphNode(id=f"tr{t}", kind="transformer",
                               label=f"配变#{t}", meta={"capacity_mw": 0.2}))
    for c in range(DEFAULT_CS_COUNT):
        nodes.append(GraphNode(id=f"cs{c}", kind="cs",
                               label=f"CS#{c}", meta={"ports": 1}))
    for c in range(min(DEFAULT_CS_COUNT, 8)):
        nodes.append(GraphNode(id=f"ev{c}", kind="ev",
                               label=f"EV#{c}", meta={"soc": round(float(_soc_curve(_time_axis(), 'gnn')[c * 14]), 3)}))
    edges: list[GraphEdge] = []
    for t in range(DEFAULT_TRANSFORMERS):
        edges.append(GraphEdge(source="env", target=f"tr{t}", kind="env-tr"))
    for c in range(DEFAULT_CS_COUNT):
        t = c % DEFAULT_TRANSFORMERS
        edges.append(GraphEdge(source=f"tr{t}", target=f"cs{c}", kind="tr-cs"))
    for c in range(min(DEFAULT_CS_COUNT, 8)):
        edges.append(GraphEdge(source=f"cs{c}", target=f"ev{c}", kind="cs-ev"))
    return HeteroGraph(nodes=nodes, edges=edges)


@dataclass
class DemoBundle:
    summary: SummaryResponse
    timeseries: TimeseriesResponse
    actions: ActionsResponse


def build_demo(device: DeviceInfo) -> DemoBundle:
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()

    hours = _time_axis()
    price = _price_curve(hours)
    arrival = _ev_arrival_profile(hours)
    online_evs = int(round(float(np.mean(arrival)) * DEFAULT_CS_COUNT * 4.2))
    gnn_power = _power_curve(hours, "gnn")
    total_power_mw = float(gnn_power[-1])
    grid_load_mw = float(gnn_power[-1] * 1.05)
    tload = _transformer_load(hours, "gnn")
    overload_rate = float(np.mean(tload > 1.0))
    # 套利收益：gnn 相对 unordered 的电费节省
    cost_gnn = float(np.sum(gnn_power * 1000.0 * price) * TIMESCALE_MIN / 60.0)
    cost_unordered = float(np.sum(_power_curve(hours, "unordered") * 1000.0 * price) * TIMESCALE_MIN / 60.0)
    arbitrage = round(cost_unordered - cost_gnn, 2)

    strategies = [_strategy_stats(s) for s in STRATEGIES]

    summary = SummaryResponse(
        online_evs=online_evs,
        total_power_mw=round(total_power_mw, 3),
        grid_load_mw=round(grid_load_mw, 3),
        transformer_overload_rate=round(overload_rate, 4),
        arbitrage_profit=arbitrage,
        device=device,
        demo_mode=True,
        data_source="demo(EV2Gym 112步对齐合成数据)",
        strategies=strategies,
        ts=ts,
    )

    power = {s: _power_curve(hours, s).tolist() for s in STRATEGIES}
    transformer_load = {s: _transformer_load(hours, s).tolist() for s in STRATEGIES}
    soc_mean = {s: _soc_curve(hours, s).tolist() for s in STRATEGIES}

    timeseries = TimeseriesResponse(
        steps=list(range(SIMULATION_LENGTH)),
        price=[round(float(p), 4) for p in price],
        power={k: [round(float(v), 4) for v in vs] for k, vs in power.items()},
        transformer_load={k: [round(float(v), 4) for v in vs] for k, vs in transformer_load.items()},
        soc_mean={k: [round(float(v), 4) for v in vs] for k, vs in soc_mean.items()},
        graph=_hetero_graph(),
        demo_mode=True,
        ts=ts,
    )

    actions: list[ActionLog] = []
    rng = np.random.default_rng(42)
    for step in range(min(SIMULATION_LENGTH, 60)):
        cs_id = int(rng.integers(0, DEFAULT_CS_COUNT))
        ev_id = int(rng.integers(0, 8))
        action_kw = round(float(rng.uniform(-22.0, 22.0)), 2)
        if price[step] > 1.0:
            reason = "峰时降功率/反向放电"
        elif price[step] < 0.5:
            reason = "谷时满功率充电"
        else:
            reason = "平时跟踪配变容量"
        actions.append(ActionLog(step=step, cs_id=cs_id, ev_id=ev_id,
                                  action_kw=action_kw, reason=reason))

    return DemoBundle(summary=summary, timeseries=timeseries,
                      actions=ActionsResponse(actions=actions, demo_mode=True, ts=ts))
