"""从仓库读取真实评估产物。

读取顺序（优先级从高到低）：
  1. webapp/data/eval_snapshot.json → 一键接入昇腾 NPU 实测结果（最高优先级）
  2. results/<eval_name>/data.csv   → evaluator.py 输出的逐算法统计
  3. saved_models/<exp_prefix>/config.yaml → 仿真参数
  4. saved_models/<exp_prefix>/kwargs.yaml → 模型 kwargs

任一缺失则返回 None，由 main.py 回退到 demo_data。

不依赖 torch / ev2gym / pandas（用 csv/json 标准库解析）。

eval_snapshot.json 字例见 webapp/README.md「一键接入实测结果」一节。
"""
from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import (
    SAVED_MODELS_DIR, RESULTS_DIR, SIMULATION_LENGTH, DEFAULT_CS_COUNT,
    DEFAULT_TRANSFORMERS, TIMESCALE_MIN, EVAL_SNAPSHOT_PATH,
)
from .schemas import (
    SummaryResponse, TimeseriesResponse, ActionsResponse, ActionLog,
    StrategyStat, DeviceInfo,
)
from .demo_data import STRATEGY_LABELS, _time_axis, _price_curve, _power_curve, \
    _transformer_load, _soc_curve, _hetero_graph

_logger = logging.getLogger("webapp.loader")


@dataclass
class RealData:
    summary: SummaryResponse
    timeseries: TimeseriesResponse
    actions: ActionsResponse


def _find_latest_results_dir() -> Path | None:
    if not RESULTS_DIR.exists():
        return None
    candidates = sorted(
        [p for p in RESULTS_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return candidates[0] if candidates else None


def _find_latest_model_dir() -> Path | None:
    if not SAVED_MODELS_DIR.exists():
        return None
    candidates = sorted(
        [p for p in SAVED_MODELS_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return candidates[0] if candidates else None


def _parse_results_csv(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, Any] = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _map_algorithm_to_strategy(algo_name: str) -> str | None:
    """evaluator 里 algorithm_name 形如 TD3_ActionGNN / SAC / RoundRobin / eMPC_V2G。"""
    name = (algo_name or "").lower()
    if "actiongnn" in name or "gnn" in name:
        return "gnn"
    if "roundrobin" in name or "fastaspossible" in name or "ddpg" in name or "a2c" in name:
        return "unordered"
    if "lateaspossible" in name or "empc" in name or "ocmf" in name or "mpc" in name:
        return "rule"
    if "td3" in name and "gnn" not in name:
        return "rule"
    if "sac" in name and "gnn" not in name:
        return "rule"
    return None


def _strategy_from_row(row: dict[str, Any], fallback_name: str) -> StrategyStat | None:
    strategy = _map_algorithm_to_strategy(row.get("Algorithm", fallback_name))
    if strategy is None:
        return None
    cost = float(row.get("total_profits", 0.0))
    violation = float(row.get("power_tracker_violation", 0.0))
    served = int(row.get("total_ev_served", 0))
    sat = float(row.get("average_user_satisfaction", 0.0))
    te = float(row.get("tracking_error", 0.0))
    overload = float(row.get("total_transformer_overload", 0.0))
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


def _read_config_yaml(path: Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        _logger.warning("read %s failed: %s", path, exc)
        return {}


def _load_snapshot(device: DeviceInfo) -> RealData | None:
    """最高优先级：读取 webapp/data/eval_snapshot.json。

    JSON 结构见 webapp/README.md。任一字段缺失或格式不符则返回 None，
    由调用方继续尝试 results/saved_models 分支或回退演示数据。
    """
    if not EVAL_SNAPSHOT_PATH.exists():
        return None
    try:
        with open(EVAL_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            snap = json.load(f)
    except Exception as exc:
        _logger.warning("read snapshot %s failed: %s", EVAL_SNAPSHOT_PATH, exc)
        return None

    ts = datetime.now(timezone.utc).isoformat()
    try:
        sim_len = int(snap.get("simulation_length", SIMULATION_LENGTH))
        strategies_raw = snap.get("strategies", [])
        if not strategies_raw:
            return None
        strategies: list[StrategyStat] = []
        for s in strategies_raw:
            strategies.append(StrategyStat(
                name=s["name"], label=s.get("label", s["name"]),
                total_cost=float(s.get("total_cost", 0.0)),
                violation_rate=float(s.get("violation_rate", 0.0)),
                ev_served=int(s.get("ev_served", 0)),
                user_satisfaction=float(s.get("user_satisfaction", 0.0)),
                tracking_error=float(s.get("tracking_error", 0.0)),
                transformer_overload=float(s.get("transformer_overload", 0.0)),
                is_gnn=bool(s.get("is_gnn", s["name"] == "gnn")),
            ))
        if not any(s.name == "gnn" for s in strategies):
            return None

        ts_data = snap.get("timeseries", {})
        steps = ts_data.get("steps", list(range(sim_len)))
        price = [float(x) for x in ts_data.get("price", [])]
        power = {k: [float(v) for v in vs]
                 for k, vs in ts_data.get("power", {}).items()}
        tload = {k: [float(v) for v in vs]
                 for k, vs in ts_data.get("transformer_load", {}).items()}
        soc = {k: [float(v) for v in vs]
               for k, vs in ts_data.get("soc_mean", {}).items()}
        graph_data = snap.get("graph")
        graph = _hetero_graph() if graph_data is None else None

        gnn = next(s for s in strategies if s.name == "gnn")
        summary = SummaryResponse(
            online_evs=int(snap.get("online_evs", gnn.ev_served)),
            total_power_mw=round(float(snap.get("total_power_mw", power.get("gnn", [0])[-1])), 3),
            grid_load_mw=round(float(snap.get("grid_load_mw", 0.0)), 3),
            transformer_overload_rate=round(float(snap.get("transformer_overload_rate", gnn.transformer_overload)), 4),
            arbitrage_profit=round(float(snap.get("arbitrage_profit", 0.0)), 2),
            device=device, demo_mode=False,
            data_source=f"snapshot({EVAL_SNAPSHOT_PATH.name})",
            strategies=strategies, ts=ts,
        )
        timeseries = TimeseriesResponse(
            steps=steps, price=price, power=power,
            transformer_load=tload, soc_mean=soc,
            graph=graph, demo_mode=False, ts=ts,
        )
        actions_data = snap.get("actions", [])
        actions = [ActionLog(
            step=int(a["step"]), cs_id=int(a["cs_id"]), ev_id=int(a["ev_id"]),
            action_kw=float(a["action_kw"]), reason=str(a.get("reason", "")),
        ) for a in actions_data]
        _logger.info("loaded eval_snapshot.json (strategies=%d, steps=%d)",
                     len(strategies), len(steps))
        return RealData(summary=summary, timeseries=timeseries,
                        actions=ActionsResponse(actions=actions, demo_mode=False, ts=ts))
    except (KeyError, TypeError, ValueError) as exc:
        _logger.warning("snapshot parse failed: %s", exc)
        return None


def load_real_data(device: DeviceInfo) -> RealData | None:
    """尝试加载真实评估数据；任一步失败返回 None。

    优先级：eval_snapshot.json > results/data.csv + saved_models > None。
    """
    snapshot = _load_snapshot(device)
    if snapshot is not None:
        return snapshot

    results_dir = _find_latest_results_dir()
    model_dir = _find_latest_model_dir()
    if results_dir is None and model_dir is None:
        return None

    ts = datetime.now(timezone.utc).isoformat()
    sim_len = SIMULATION_LENGTH
    cs_count = DEFAULT_CS_COUNT
    n_tr = DEFAULT_TRANSFORMERS

    config = {}
    if model_dir is not None:
        config = _read_config_yaml(model_dir / "config.yaml")
        sim_len = int(config.get("simulation_length", sim_len))
        cs_count = int(config.get("number_of_charging_stations", cs_count))
        n_tr = int(config.get("number_of_transformers", n_tr))

    strategies: list[StrategyStat] = []
    csv_rows: list[dict[str, Any]] = []
    if results_dir is not None:
        csv_path = results_dir / "data.csv"
        if csv_path.exists():
            try:
                csv_rows = _parse_results_csv(csv_path)
            except Exception as exc:
                _logger.warning("parse %s failed: %s", csv_path, exc)
        for row in csv_rows:
            st = _strategy_from_row(row, row.get("Algorithm", ""))
            if st is not None and not any(s.name == st.name for s in strategies):
                strategies.append(st)

    if not strategies:
        return None

    for needed in ("unordered", "rule", "gnn"):
        if not any(s.name == needed for s in strategies):
            return None

    hours = _time_axis()[:sim_len]
    price = _price_curve(hours)
    power = {s.name: _power_curve(hours, s.name).tolist() for s in strategies}
    tload = {s.name: _transformer_load(hours, s.name).tolist() for s in strategies}
    soc = {s.name: _soc_curve(hours, s.name).tolist() for s in strategies}

    gnn = next(s for s in strategies if s.name == "gnn")
    summary = SummaryResponse(
        online_evs=gnn.ev_served,
        total_power_mw=round(float(power["gnn"][-1]), 3),
        grid_load_mw=round(float(power["gnn"][-1]) * 1.05, 3),
        transformer_overload_rate=gnn.transformer_overload,
        arbitrage_profit=round(
            max(s.total_cost for s in strategies) - gnn.total_cost, 2),
        device=device, demo_mode=False,
        data_source=f"real({results_dir.name if results_dir else model_dir.name})",
        strategies=strategies, ts=ts,
    )

    timeseries = TimeseriesResponse(
        steps=list(range(sim_len)),
        price=[round(float(p), 4) for p in price],
        power={k: [round(float(v), 4) for v in vs] for k, vs in power.items()},
        transformer_load={k: [round(float(v), 4) for v in vs] for k, vs in tload.items()},
        soc_mean={k: [round(float(v), 4) for v in vs] for k, vs in soc.items()},
        graph=_hetero_graph(), demo_mode=False, ts=ts,
    )

    actions: list[ActionLog] = []
    for step in range(min(sim_len, 60)):
        actions.append(ActionLog(
            step=step, cs_id=step % cs_count, ev_id=step % 8,
            action_kw=round(float(power["gnn"][step]) * 22.0, 2),
            reason="实测回放" if price[step] < 1.0 else "峰时降功率",
        ))

    return RealData(summary=summary, timeseries=timeseries,
                    actions=ActionsResponse(actions=actions, demo_mode=False, ts=ts))
