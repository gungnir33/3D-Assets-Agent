from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from local_3d_agent.metadata import JobMetadata, create_job
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
            with app.state.gpu_lock:
                output = getattr(active_backend(), f"generate_{kind}")(payload, job)
            metadata.finish(output)
            return GenerationResponse(job_id=job.name, file=str(output), type=Path(output).suffix.lstrip("."),
                                      metadata=str(job / "metadata.json"))
        except ApiError:
            raise
        except Exception as exc:
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
