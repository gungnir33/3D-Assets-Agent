from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image

from local_3d_agent.service.backend import GenerationBackend


def test_text_generation_releases_shape_and_paint_before_t2i(tmp_path: Path) -> None:
    events = []
    service = GenerationBackend.__new__(GenerationBackend)
    service.manager = SimpleNamespace(
        release=lambda name: events.append(f"release:{name}"),
        cleanup_cuda=lambda: events.append("cleanup"),
        acquire_t2i=lambda: object(),
    )
    service._resolve = lambda: None
    def fake_shape(image, job, request):
        events.append("shape")
        output = job / "raw_mesh.glb"
        output.write_bytes(b"mesh")
        return output
    service._shape = fake_shape
    request = SimpleNamespace(prompt="robot", seed=1, texture=False)
    from local_3d_agent.pipelines import text_to_image
    original = text_to_image.TextToImagePipeline.generate
    text_to_image.TextToImagePipeline.generate = lambda self, prompt, output, seed: events.append("t2i") or output
    try:
        service.generate_text(request, tmp_path)
    finally:
        text_to_image.TextToImagePipeline.generate = original
    assert events[:4] == ["release:shape", "release:paint", "cleanup", "t2i"]


def test_paint_releases_shape_before_loading_paint(tmp_path: Path, monkeypatch) -> None:
    events = []
    service = GenerationBackend.__new__(GenerationBackend)
    service.manager = SimpleNamespace(
        release=lambda name: events.append(f"release:{name}"),
        cleanup_cuda=lambda: events.append("cleanup"),
        acquire_paint=lambda: events.append("acquire:paint") or object(),
    )
    monkeypatch.setattr("local_3d_agent.service.backend.trimesh.load", lambda *a, **k: object())
    monkeypatch.setattr("local_3d_agent.service.backend.TexturePipeline.generate", lambda *a, **k: tmp_path / "model.glb")
    request = SimpleNamespace(face_count=40000)
    service._paint(tmp_path / "raw.glb", tmp_path / "condition.png", tmp_path, request)
    assert events == ["release:shape", "release:t2i", "cleanup", "acquire:paint"]


def test_opaque_input_fails_clearly_when_rembg_is_missing_and_download_disabled(tmp_path: Path) -> None:
    source = tmp_path / "opaque.jpg"
    Image.new("RGB", (8, 8), "white").save(source)
    service = GenerationBackend.__new__(GenerationBackend)
    service.settings = SimpleNamespace(
        paths=SimpleNamespace(rembg_model_dir=tmp_path / "rembg"),
        models=SimpleNamespace(allow_download=False),
    )
    job = tmp_path / "job"
    job.mkdir()

    with pytest.raises(RuntimeError, match="rembg model is missing"):
        service._copy_input(source, job)


def test_transparent_input_does_not_require_rembg(tmp_path: Path) -> None:
    source = tmp_path / "transparent.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(source)
    job = tmp_path / "job"
    job.mkdir()
    service = GenerationBackend.__new__(GenerationBackend)
    service.settings = SimpleNamespace(
        paths=SimpleNamespace(rembg_model_dir=tmp_path / "rembg"),
        models=SimpleNamespace(allow_download=False),
    )

    assert service._copy_input(source, job).is_file()
