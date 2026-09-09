import json
import shutil
from pathlib import Path
import httpx

BASE_URL = "http://127.0.0.1:8080"
TIMEOUT_SECONDS = 900.0
START_COMMAND = "/home/mcl/workspace/3D-Assets-Agent/scripts/start_server.sh"

def check_health() -> bool:
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=10)
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        print(f"LOCAL_3D_SERVER_NOT_RUNNING: start with {START_COMMAND}")
        return False

def post_generation(endpoint: str, payload: dict, output_dir: str | None = None) -> int:
    if not check_health():
        return 2
    try:
        response = httpx.post(f"{BASE_URL}/generate/{endpoint}", json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(json.dumps({"error": "LOCAL_3D_API_ERROR", "message": str(exc)}))
        return 1
    result = response.json()
    if output_dir:
        target_dir = Path(output_dir).expanduser().resolve(); target_dir.mkdir(parents=True, exist_ok=True)
        source = Path(result["file"]); target = target_dir / source.name; shutil.copy2(source, target); result["file"] = str(target)
    print(json.dumps(result, ensure_ascii=False))
    return 0

