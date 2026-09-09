# Design summary

V0.1 is a localhost-only synchronous FastAPI service with one GPU lock. A central manifest/resolver validates fixed local model paths and optionally fills missing models from official repositories. Inference always receives local paths.

Text uses HunyuanDiT for a retained condition image, releases it, then runs Hunyuan3D Shape. Image skips HunyuanDiT. Every mesh sent to Paint passes validation, FloaterRemover, DegenerateFaceRemover, and FaceReducer (default 40,000 faces). GLB validation requires reloadable nonempty finite geometry; textured output additionally requires UV, material, and an actual embedded texture image.

Each job keeps reproducibility metadata, intermediate meshes/images, timings, model paths/revisions, download state, and peak VRAM. The Skill is a thin HTTP client and does not own server/model/environment lifecycle. Model weights, generated/private assets, caches, tokens, `.env`, and upstream source are never committed.
