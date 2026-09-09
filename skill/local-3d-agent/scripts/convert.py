#!/usr/bin/env python3
import argparse
from pathlib import Path
import trimesh
p=argparse.ArgumentParser(); p.add_argument("input"); p.add_argument("output"); a=p.parse_args()
source=Path(a.input).expanduser().resolve(); target=Path(a.output).expanduser().resolve()
if target.suffix.lower()==".fbx": raise SystemExit("FBX conversion requires Blender and is not enabled by this helper")
mesh=trimesh.load(source, force="mesh"); target.parent.mkdir(parents=True,exist_ok=True); mesh.export(target); print(target)
