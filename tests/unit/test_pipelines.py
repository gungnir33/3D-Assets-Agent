from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from local_3d_agent.pipelines.image_to_3d import ImageTo3DPipeline
from local_3d_agent.pipelines.text_to_3d import TextTo3DPipeline
from local_3d_agent.pipelines.text_to_image import TextToImagePipeline


class ExportableMesh:
    def export(self, path):
        Path(path).write_bytes(b"mesh")


def test_image_pipeline_passes_seed_and_shape_steps(tmp_path: Path) -> None:
    calls = []
    backend = lambda **kwargs: calls.append(kwargs) or [ExportableMesh()]
    pipeline = ImageTo3DPipeline(backend, generator_factory=lambda seed: f"generator-{seed}")
    source = tmp_path / "input.png"
    Image.new("RGB", (4, 4), "white").save(source)

    output = pipeline.generate(source, tmp_path / "raw.glb", seed=42, shape_steps=17)

    assert output.read_bytes() == b"mesh"
    assert calls[0]["num_inference_steps"] == 17
    assert calls[0]["generator"] == "generator-42"


def test_text_to_image_saves_condition_image_with_seed(tmp_path: Path) -> None:
    calls = []
    backend = lambda prompt, seed: calls.append((prompt, seed)) or Image.new("RGB", (4, 4), "white")

    output = TextToImagePipeline(backend).generate("robot", tmp_path / "condition.png", seed=9)

    assert output.is_file()
    assert calls == [("robot", 9)]


def test_text_to_3d_releases_t2i_before_shape(tmp_path: Path) -> None:
    events = []
    t2i = SimpleNamespace(generate=lambda prompt, output, seed: events.append("t2i") or output)
    shape = SimpleNamespace(generate=lambda image, output, seed, shape_steps: events.append("shape") or output)
    manager = SimpleNamespace(release=lambda name: events.append(f"release:{name}"), cleanup_cuda=lambda: events.append("cleanup"))
    pipeline = TextTo3DPipeline(t2i=t2i, shape=shape, model_manager=manager)

    result = pipeline.generate("robot", tmp_path, seed=5, shape_steps=20)

    assert result == tmp_path / "raw_mesh.glb"
    assert events == ["t2i", "release:t2i", "cleanup", "shape"]

