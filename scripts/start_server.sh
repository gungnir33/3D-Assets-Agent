#!/usr/bin/env bash
set -euo pipefail
export HUNYUAN3D_SOURCE_ROOT="${HUNYUAN3D_SOURCE_ROOT:-/home/mcl/workspace/hunyuan3d/Hunyuan3D-2}"
export U2NET_HOME="${U2NET_HOME:-/home/mcl/models/rembg}"
exec /home/mcl/anaconda3/bin/conda run --no-capture-output -n hunyuan3d \
  python -m uvicorn local_3d_agent.service.app:app --host 127.0.0.1 --port 8080
