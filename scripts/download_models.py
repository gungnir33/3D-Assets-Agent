#!/usr/bin/env python3
"""Explicitly fill missing models at manifest-defined locations."""
import argparse
from pathlib import Path
from local_3d_agent.model_resolver import load_manifest, resolve_model

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=("hunyuan3d", "hunyuandit"))
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    specs = load_manifest(ROOT / "config/model_manifest.yaml")
    names = ("hunyuan3d", "hunyuandit") if args.all else (args.model,)
    for name in names:
        resolved = resolve_model(specs[name], allow_download=True)
        print(f"{name}: READY path={resolved.local_path} downloaded={str(resolved.downloaded).lower()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
