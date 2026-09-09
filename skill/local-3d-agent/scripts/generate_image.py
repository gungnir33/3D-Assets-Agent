#!/usr/bin/env python3
import argparse
from pathlib import Path
from client import post_generation
p=argparse.ArgumentParser(); p.add_argument("image"); p.add_argument("--output-dir"); p.add_argument("--seed",type=int,default=12345); p.add_argument("--face-count",type=int,default=40000); p.add_argument("--shape-steps",type=int,default=50); p.add_argument("--no-texture",action="store_true"); a=p.parse_args()
raise SystemExit(post_generation("image", {"image":str(Path(a.image).expanduser().resolve()),"seed":a.seed,"face_count":a.face_count,"shape_steps":a.shape_steps,"texture":not a.no_texture,"format":"glb"}, a.output_dir))

