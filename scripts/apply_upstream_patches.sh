#!/usr/bin/env bash
set -euo pipefail
source_root="${HUNYUAN3D_SOURCE_ROOT:-/home/mcl/workspace/hunyuan3d/Hunyuan3D-2}"
target="$source_root/hy3dgen/texgen/utils/multiview_utils.py"
if grep -q 'trust_remote_code=True' "$target"; then
  echo "Hunyuan3D local custom pipeline patch already applied"
  exit 0
fi
python - "$target" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
old = "custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16)"
new = "custom_pipeline=custom_pipeline_path,\n            trust_remote_code=True,\n            torch_dtype=torch.float16)"
text = path.read_text()
if old not in text:
    raise SystemExit("expected upstream code not found; refusing an unsafe patch")
path.write_text(text.replace(old, new, 1))
PY
echo "Applied local custom pipeline compatibility patch"
