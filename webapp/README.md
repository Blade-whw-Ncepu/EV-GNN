# 智充图驭 — Web 驾驶舱

大规模电动汽车智能充电调度云平台可视化大盘。基于华为云应用交付架构，FastAPI 后端 + 单页前端，零构建、零 torch 依赖，本地与 ECS 均可一键启动。

## 快速启动

```bash
cd webapp
pip install -r requirements.txt
bash run.sh                 # 默认 8000 端口
# 或
bash run.sh 9000
```

浏览器打开 `http://localhost:8000`。

## 环境变量

| 变量 | 说明 | 默认 |
|---|---|---|
| `WEBAPP_PORT` | 监听端口 | 8000 |
| `EVGNN_REPO_ROOT` | 仓库根目录（用于读取 saved_models/results 与设备探测） | 自动推断为 webapp 上级目录 |
| `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DB` | 华为云 RDS MySQL；未配置则降级本地 SQLite | 未配置 |
| `REDIS_URL` | 华为云 DCS Redis；未配置则降级进程内 LRU | 未配置 |
| `WEBAPP_DEVICE_FORCE` | 强制设备标识（npu:0 / cpu），调试用 | 自动探测 |

## 接口

- `GET /api/health` — 华为云探活
- `GET /api/dashboard/summary` — 顶部 KPI 与策略对比
- `GET /api/dashboard/timeseries` — 112 步功率/电价/配变负载曲线 + 异构图
- `GET /api/dashboard/actions` — 实时调度动作日志
- `POST /api/dashboard/diagnosis` — AI 调度诊断摘要（预留，待接入华为云 MaaS）

## 数据模式

- 优先读取 `webapp/data/eval_snapshot.json`（最高优先级，`demo_mode=false`）。
- 其次读取仓库 `saved_models/` 与 `results/` 真实评估产物（`demo_mode=false`）。
- 缺失时自动回退为与 EV2Gym 112 步仿真对齐的结构化演示数据（`demo_mode=true`，前端显式标注「演示」）。

## 一键接入实测结果

将昇腾 NPU（或 CPU）上跑完的评估结果写入 `webapp/data/eval_snapshot.json`，重启服务即自动以实测数据驱动大盘（`demo_mode=false`）。字段示例：

```json
{
  "online_evs": 88,
  "total_power_mw": 0.412,
  "grid_load_mw": 0.433,
  "transformer_overload_rate": 0.0268,
  "arbitrage_profit": 1284.5,
  "simulation_length": 112,
  "strategies": [
    {"name": "unordered", "label": "无序充电", "total_cost": 9421.3, "violation_rate": 0.125,
     "ev_served": 82, "user_satisfaction": 0.721, "tracking_error": 0.312, "transformer_overload": 1.84, "is_gnn": false},
    {"name": "rule",      "label": "规则控制", "total_cost": 8102.7, "violation_rate": 0.054,
     "ev_served": 85, "user_satisfaction": 0.858, "tracking_error": 0.183, "transformer_overload": 0.62, "is_gnn": false},
    {"name": "gnn",       "label": "EV-GNN(Ascend NPU)", "total_cost": 7136.8, "violation_rate": 0.018,
     "ev_served": 88, "user_satisfaction": 0.932, "tracking_error": 0.069, "transformer_overload": 0.21, "is_gnn": true}
  ],
  "timeseries": {
    "steps": [0, 1, 2, "...", 111],
    "price": [0.35, 0.35, "...", 1.25],
    "power":          {"unordered": [0.51, "..."], "rule": [0.43, "..."], "gnn": [0.39, "..."]},
    "transformer_load": {"unordered": [0.85, "..."], "rule": [0.72, "..."], "gnn": [0.65, "..."]},
    "soc_mean":       {"unordered": [0.31, "..."], "rule": [0.38, "..."], "gnn": [0.42, "..."]}
  },
  "actions": [
    {"step": 0, "cs_id": 3, "ev_id": 1, "action_kw": 22.0, "reason": "谷时满功率充电"},
    {"step": 48, "cs_id": 7, "ev_id": 2, "action_kw": -11.0, "reason": "峰时反向放电"}
  ]
}
```

`graph` 字段可省略，缺省时由演示数据生成器补齐异构图骨架。

## 华为云部署

```bash
# ECS 上
export MYSQL_HOST=xxx.mysql.rds.huaweicloud.com
export MYSQL_USER=evgnn MYSQL_PASSWORD=xxx MYSQL_DB=evgnn
export REDIS_URL=redis://xxx.dcs.huaweicloud.com:6379/0
bash run.sh
```

未配置云实例时自动降级为本地 SQLite + 内存 LRU，零报错启动。
