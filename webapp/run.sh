#!/usr/bin/env bash
# 智充图驭 Web 驾驶舱启动脚本
# 用法: bash run.sh [port]   默认端口 8000
set -euo pipefail
PORT="${1:-${WEBAPP_PORT:-8000}}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PYTHONUNBUFFERED=1
exec python -m uvicorn backend.main:app --host 0.0.0.0 --port "$PORT"
