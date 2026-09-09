---
name: local-3d-agent
description: Use when a user asks to generate a local 3D asset from text or an image, texture an existing mesh, inspect the local Hunyuan3D service, or convert a generated GLB/OBJ/FBX asset.
---

# Local 3D Agent

Use the local FastAPI service for generation. The server owns CUDA models, model resolution, downloads, and the Conda environment; this Skill is only a local HTTP client.

## Workflow

1. Run `python scripts/health.py` before generation.
2. If it reports `LOCAL_3D_SERVER_NOT_RUNNING`, tell the user to run `/home/mcl/workspace/3D-Assets-Agent/scripts/start_server.sh`. Do not start or daemonize it yourself.
3. Select exactly one command:
   - Text: `python scripts/generate_text.py PROMPT --output-dir assets`
   - Image: `python scripts/generate_image.py /absolute/input.png --output-dir assets`
   - Texture: `python scripts/texture.py /absolute/mesh.glb /absolute/condition.png --output-dir assets`
   - Convert: `python scripts/convert.py INPUT OUTPUT`
4. Return the absolute generated file and metadata paths reported by the script.

Generation requests may take 15 minutes or longer. Do not shorten the client timeout, directly import Hunyuan3D/Torch, download model weights, or modify Conda from this Skill.

Read [references/workflow.md](references/workflow.md) when choosing generation parameters or troubleshooting an API error.

## Quick reference

| Request | Script |
|---|---|
| Service status | `health.py` |
| Text → 3D | `generate_text.py` |
| Image → 3D | `generate_image.py` |
| Existing mesh → texture | `texture.py` |
| Local format conversion | `convert.py` |

