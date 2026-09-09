#!/usr/bin/env python3
import argparse
from pathlib import Path
from client import post_generation
p=argparse.ArgumentParser(); p.add_argument("mesh"); p.add_argument("condition_image"); p.add_argument("--output-dir"); p.add_argument("--seed",type=int,default=12345); p.add_argument("--face-count",type=int,default=40000); a=p.parse_args()
raise SystemExit(post_generation("texture", {"mesh":str(Path(a.mesh).resolve()),"condition_image":str(Path(a.condition_image).resolve()),"seed":a.seed,"face_count":a.face_count,"texture":True,"format":"glb","shape_steps":50}, a.output_dir))

