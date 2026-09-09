# Deployment

The required runtime is Conda `hunyuan3d`, Python 3.10. Upstream source must be an independent checkout at `/home/mcl/workspace/hunyuan3d/Hunyuan3D-2`; record its exact SHA with `git rev-parse HEAD`.

Compile extensions from the upstream checkout:

```bash
conda run --no-capture-output -n hunyuan3d python hy3dgen/texgen/custom_rasterizer/setup.py install
conda run --no-capture-output -n hunyuan3d python hy3dgen/texgen/differentiable_renderer/setup.py install
```

Run ordinary verification with `conda run -n hunyuan3d python -m pytest tests -q`. GPU E2E is intentionally opt-in: `RUN_GPU_E2E=1 conda run -n hunyuan3d python -m pytest tests/gpu -m gpu -v`. It executes large models sequentially and requires sufficient free VRAM.

Only declare deployment complete after Shape, HunyuanDiT, Paint, all three E2E flows, textured GLB inspection, live API, Skill call, commit, and push all have fresh evidence.

