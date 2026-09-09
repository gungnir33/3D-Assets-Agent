#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import httpx

def main():
    p=argparse.ArgumentParser(); p.add_argument("--image",type=Path); p.add_argument("--prompt"); a=p.parse_args()
    health=httpx.get("http://127.0.0.1:8080/health",timeout=10); health.raise_for_status(); print(json.dumps(health.json()))
    if a.image:
        response=httpx.post("http://127.0.0.1:8080/generate/image",json={"image":str(a.image.resolve()),"texture":True},timeout=900)
    elif a.prompt:
        response=httpx.post("http://127.0.0.1:8080/generate/text",json={"prompt":a.prompt,"texture":True},timeout=900)
    else:
        return 0
    response.raise_for_status(); print(json.dumps(response.json())); return 0

if __name__ == "__main__": raise SystemExit(main())
