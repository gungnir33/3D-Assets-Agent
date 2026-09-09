#!/usr/bin/env python3
"""Report local model readiness without downloading anything."""
from pathlib import Path
from local_3d_agent.model_resolver import load_manifest, validate_model

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    failed = False
    for name, spec in load_manifest(ROOT / "config/model_manifest.yaml").items():
        result = validate_model(spec)
        print(f"{name}: {result.status.value}" + (f" ({', '.join(result.missing)})" if result.missing else ""))
        failed |= result.status.value != "READY"
    return int(failed)

if __name__ == "__main__":
    raise SystemExit(main())
