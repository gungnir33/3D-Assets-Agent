from pathlib import Path

from local_3d_agent.pipelines.texture import TexturePipeline


class Mesh:
    def export(self, path):
        Path(path).write_bytes(b"mesh")


def test_texture_pipeline_preprocesses_before_paint(tmp_path: Path) -> None:
    events = []
    raw = Mesh()
    processed = Mesh()
    textured = Mesh()
    pipeline = TexturePipeline(
        backend=lambda mesh, image: events.append("paint") or textured,
        preprocessor=lambda mesh, faces: events.append(f"preprocess:{faces}") or processed,
        validator=lambda path, require_texture=False: events.append(f"validate:{require_texture}"),
    )

    result = pipeline.generate(raw, tmp_path / "condition.png", tmp_path, face_count=40000)

    assert result == tmp_path / "model.glb"
    assert (tmp_path / "processed_mesh.glb").is_file()
    assert events == ["preprocess:40000", "paint", "validate:True"]

