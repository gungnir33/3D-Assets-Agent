from __future__ import annotations
import time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from local_3d_agent.metadata import JobMetadata, create_job
from local_3d_agent.mesh.export import export_mesh
from .errors import ApiError
from .gpu_lock import GpuTaskLock
from .schemas import GenerationResponse, ImageRequest, TextRequest, TextureRequest

def build_default_app() -> FastAPI:
    from local_3d_agent.config import DEFAULT_CONFIG, load_settings
    from .backend import GenerationBackend
    settings = load_settings(DEFAULT_CONFIG)
    root = Path(__file__).resolve().parents[3]
    return create_app(settings, backend=GenerationBackend(settings, root / "config/model_manifest.yaml"))

def create_app(settings, *, backend=None) -> FastAPI:
    app = FastAPI(title="Local 3D Assets Agent", version="0.1.0")
    app.state.settings = settings
    app.state.backend = backend
    app.state.gpu_lock = GpuTaskLock()

    @app.exception_handler(ApiError)
    def api_error_handler(_: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": {"code": exc.code, "message": str(exc), "details": exc.details}})

    def active_backend():
        if app.state.backend is None:
            raise ApiError("BACKEND_NOT_CONFIGURED", "generation backend is not configured")
        return app.state.backend

    @app.get("/health")
    def health():
        return active_backend().health()

    def run(kind: str, payload):
        job = create_job(settings.paths.assets)
        metadata = JobMetadata(job, seed=payload.seed, prompt=getattr(payload, "prompt", None),
                               shape_steps=payload.shape_steps, face_count=payload.face_count,
                               device=settings.runtime.device, dtype=settings.runtime.dtype)
        try:
            started = time.monotonic()
            peak_vram = 0
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except (ImportError, RuntimeError):
                torch = None
            with app.state.gpu_lock:
                output = getattr(active_backend(), f"generate_{kind}")(payload, job)
            output = Path(output)
            if payload.format != output.suffix.lstrip(".").lower():
                import trimesh
                mesh = trimesh.load(output, force="mesh")
                output = export_mesh(mesh, job / "model", payload.format)
            if torch is not None and torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated()
            metadata.record_stage("generation", time.monotonic() - started, peak_vram=peak_vram)
            resolved_models = getattr(active_backend(), "resolved_models", lambda: {})()
            for name, model in resolved_models.items():
                metadata.record_model(name, model["path"], model.get("revision"),
                                      downloaded=model.get("downloaded", False))
            if kind == "text":
                metadata.record_inputs(condition_image=str(job / "condition.png"))
            elif kind == "image":
                copied = str(job / "input.png")
                metadata.record_inputs(input_image=copied, condition_image=copied)
            else:
                metadata.record_inputs(condition_image=str(payload.condition_image),
                                       input_mesh=str(payload.mesh))
            metadata.finish(output)
            return GenerationResponse(job_id=job.name, file=str(output), type=Path(output).suffix.lstrip("."),
                                      metadata=str(job / "metadata.json"))
        except ApiError:
            raise
        except Exception as exc:
            try:
                import torch
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError)
            except ImportError:
                torch, is_oom = None, False
            if is_oom:
                details = {
                    "allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                    "reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
                    "total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
                }
                manager = getattr(app.state.backend, "manager", None)
                if manager is not None:
                    manager.release_all()
                else:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                metadata.fail("CUDA_OUT_OF_MEMORY", str(exc))
                raise ApiError("CUDA_OUT_OF_MEMORY", str(exc), details=details, status_code=503) from exc
            metadata.fail("GENERATION_FAILED", str(exc))
            raise ApiError("GENERATION_FAILED", str(exc)) from exc

    @app.post("/generate/image", response_model=GenerationResponse)
    def generate_image(payload: ImageRequest): return run("image", payload)
    @app.post("/generate/text", response_model=GenerationResponse)
    def generate_text(payload: TextRequest): return run("text", payload)
    @app.post("/generate/texture", response_model=GenerationResponse)
    def generate_texture(payload: TextureRequest): return run("texture", payload)
    return app

app = build_default_app()
