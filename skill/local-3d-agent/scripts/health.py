#!/usr/bin/env python3
import json, httpx
from client import BASE_URL, check_health
if not check_health(): raise SystemExit(2)
response = httpx.get(f"{BASE_URL}/health", timeout=10); response.raise_for_status()
print(json.dumps(response.json(), ensure_ascii=False)); raise SystemExit(0)

