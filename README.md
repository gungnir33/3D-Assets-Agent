# 3D Assets Agent

Local-first Hunyuan3D asset generation service and Codex Skill for Text-to-3D, Image-to-3D, and texturing an existing mesh.

> Model weights are not included in this Git repository.

## Architecture

`Codex → local-3d-agent → 127.0.0.1:8080 → HunyuanDiT/Hunyuan3D Shape/Paint → GLB`

The project imports upstream source from `/home/mcl/workspace/hunyuan3d/Hunyuan3D-2`. Model weights are resolved locally first and downloaded from manifest-declared official repositories only when incomplete and `allow_download` is enabled.

## Requirements and setup

- Ubuntu, NVIDIA GPU with at least 24.5 GB VRAM (validated target: RTX 5880 Ada 48 GB)
- Conda environment `hunyuan3d`, Python 3.10
- Model roots shown in `config/default.yaml`; override with `.env`/`LOCAL_3D_*`

```bash
conda create -n hunyuan3d python=3.10 -y
conda run -n hunyuan3d python -m pip install -e '.[test]'
conda run -n hunyuan3d python scripts/check_models.py
conda run -n hunyuan3d python scripts/check_environment.py
```

Missing models may be explicitly filled into fixed paths with `scripts/download_models.py --model NAME` or `--all`. The project respects `HF_ENDPOINT`, `HTTP_PROXY`, and `HTTPS_PROXY` and never overrides them.

## Server and API

Start the foreground localhost-only service:

```bash
./scripts/start_server.sh
curl http://127.0.0.1:8080/health
```

```bash
curl -X POST http://127.0.0.1:8080/generate/text -H 'Content-Type: application/json' -d '{"prompt":"industrial inspection robot","texture":true,"seed":12345,"format":"glb","face_count":40000,"shape_steps":50}'
curl -X POST http://127.0.0.1:8080/generate/image -H 'Content-Type: application/json' -d '{"image":"/absolute/input.png","texture":true}'
curl -X POST http://127.0.0.1:8080/generate/texture -H 'Content-Type: application/json' -d '{"mesh":"/absolute/model.glb","condition_image":"/absolute/input.png"}'
```

Requests are synchronous and serialized by one GPU lock. Generated jobs and metadata are stored under `assets/` and excluded from Git.

## Codex Skill

```bash
mkdir -p /home/mcl/.agents/skills
ln -sfn /home/mcl/workspace/3D-Assets-Agent/skill/local-3d-agent /home/mcl/.agents/skills/local-3d-agent
```

The Skill checks health first and never loads models or starts a daemon. See `skill/local-3d-agent/references/workflow.md`.

## Troubleshooting

- `LOCAL_3D_SERVER_NOT_RUNNING`: run `scripts/start_server.sh` in a terminal.
- Model `MISSING`/`PARTIAL`: inspect `scripts/check_models.py`, then explicitly allow/fill missing weights.
- CUDA OOM: stop only GPU workloads you own, then retry. The service records memory metrics and cleans its pipelines; it never kills user processes.
- FBX: optional and requires Blender; GLB is the supported primary format.
- `libc10.so` importing an extension: import Torch before `custom_rasterizer_kernel`/`mesh_processor`.

API details are in `docs/api.md`; deployment and GPU checks are in `docs/deployment.md`.

## License notice

Project-owned source code is licensed under the MIT License. Hunyuan3D models and upstream code are governed by Tencent's respective licenses. HunyuanDiT is governed by its corresponding upstream license.
