from __future__ import annotations
import shutil
from pathlib import Path
import trimesh
from local_3d_agent.model_manager import ModelManager
from local_3d_agent.model_resolver import load_manifest, resolve_model, validate_model
from local_3d_agent.pipelines.image_to_3d import ImageTo3DPipeline
from local_3d_agent.pipelines.text_to_image import TextToImagePipeline
from local_3d_agent.pipelines.texture import TexturePipeline

class GenerationBackend:
    def __init__(self, settings, manifest_path: Path):
        self.settings = settings
        self.specs = load_manifest(manifest_path)
        self.manager: ModelManager | None = None
        self._resolved = {}

    def _resolve(self):
        h3d = resolve_model(self.specs["hunyuan3d"], allow_download=self.settings.models.allow_download)
        dit = resolve_model(self.specs["hunyuandit"], allow_download=self.settings.models.allow_download)
        self._resolved = {"hunyuan3d": h3d, "hunyuandit": dit}
        if self.manager is None:
            self.manager = ModelManager(self.settings.paths.hunyuan3d_source, h3d.local_path, dit.local_path,
                                        self.settings.runtime.device)
        return h3d, dit

    def resolved_models(self):
        return {
            name: {
                "path": str(model.local_path),
                "revision": model.revision,
                "downloaded": model.downloaded,
            }
            for name, model in self._resolved.items()
        }

    def health(self):
        import torch
        labels = {"hunyuan3d_shape": "hunyuan3d", "hunyuan3d_paint": "hunyuan3d", "hunyuandit": "hunyuandit"}
        models = {}
        for label, name in labels.items():
            result = validate_model(self.specs[name])
            models[label] = result.status.value.lower() if result.status.value == "READY" else (
                "downloadable" if self.settings.models.allow_download and self.specs[name].remote_repo else result.status.value.lower())
        cuda = torch.cuda.is_available()
        return {"status": "ok" if cuda else "degraded", "cuda": cuda,
                "gpu": torch.cuda.get_device_name(0) if cuda else None, "models": models}

    @staticmethod
    def _copy_input(source: Path, job: Path) -> Path:
        target = job / "input.png"
        from PIL import Image
        Image.open(source).convert("RGBA").save(target)
        return target

    def _shape(self, image: Path, job: Path, request) -> Path:
        backend = self.manager.acquire_shape()
        return ImageTo3DPipeline(backend).generate(image, job / "raw_mesh.glb", seed=request.seed,
                                                    shape_steps=request.shape_steps)

    def _paint(self, raw_path: Path, image: Path, job: Path, request) -> Path:
        mesh = trimesh.load(raw_path, force="mesh")
        self.manager.release("shape")
        self.manager.release("t2i")
        self.manager.cleanup_cuda()
        return TexturePipeline(self.manager.acquire_paint()).generate(mesh, image, job,
                                                                      face_count=request.face_count)

    def generate_image(self, request, job_dir: Path) -> Path:
        self._resolve()
        image = self._copy_input(request.image, job_dir)
        raw = self._shape(image, job_dir, request)
        if request.texture:
            return self._paint(raw, image, job_dir, request)
        output = job_dir / "model.glb"; shutil.copy2(raw, output); return output

    def generate_text(self, request, job_dir: Path) -> Path:
        self._resolve()
        self.manager.release("shape")
        self.manager.release("paint")
        self.manager.cleanup_cuda()
        condition = TextToImagePipeline(self.manager.acquire_t2i()).generate(
            request.prompt, job_dir / "condition.png", seed=request.seed)
        self.manager.release("t2i"); self.manager.cleanup_cuda()
        raw = self._shape(condition, job_dir, request)
        if request.texture:
            return self._paint(raw, condition, job_dir, request)
        output = job_dir / "model.glb"; shutil.copy2(raw, output); return output

    def generate_texture(self, request, job_dir: Path) -> Path:
        self._resolve()
        mesh = trimesh.load(request.mesh, force="mesh")
        return TexturePipeline(self.manager.acquire_paint()).generate(
            mesh, request.condition_image, job_dir, face_count=request.face_count)
