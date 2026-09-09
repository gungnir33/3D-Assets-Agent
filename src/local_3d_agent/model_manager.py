"""Lazy ownership and release of large GPU pipelines."""
from __future__ import annotations
import gc
import sys
from pathlib import Path

class ModelManager:
    def __init__(self, source_root: Path, hunyuan3d_path: Path, hunyuandit_path: Path,
                 device: str = "cuda"):
        self.source_root = source_root.resolve()
        self.hunyuan3d_path = hunyuan3d_path.resolve()
        self.hunyuandit_path = hunyuandit_path.resolve()
        self.device = device
        self.models: dict[str, object] = {}

    def _ensure_source(self) -> None:
        if not (self.source_root / "hy3dgen").is_dir():
            raise RuntimeError(f"invalid Hunyuan3D source root: {self.source_root}")
        source = str(self.source_root)
        if source not in sys.path:
            sys.path.insert(0, source)

    def acquire_shape(self):
        if "shape" not in self.models:
            self._ensure_source()
            import torch
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            self.models["shape"] = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                str(self.hunyuan3d_path), subfolder="hunyuan3d-dit-v2-0",
                variant="fp16", use_safetensors=True, device=self.device,
                dtype=torch.float16,
            )
        return self.models["shape"]

    def acquire_t2i(self):
        if "t2i" not in self.models:
            self._ensure_source()
            from hy3dgen.text2image import HunyuanDiTPipeline
            self.models["t2i"] = HunyuanDiTPipeline(str(self.hunyuandit_path), device=self.device)
        return self.models["t2i"]

    def acquire_paint(self):
        if "paint" not in self.models:
            self._ensure_source()
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            self.models["paint"] = Hunyuan3DPaintPipeline.from_pretrained(
                str(self.hunyuan3d_path), subfolder="hunyuan3d-paint-v2-0")
        return self.models["paint"]

    def release(self, name: str) -> None:
        self.models.pop(name, None)

    def release_all(self) -> None:
        self.models.clear()
        self.cleanup_cuda()

    @staticmethod
    def cleanup_cuda() -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
