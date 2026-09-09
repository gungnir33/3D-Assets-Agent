from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from local_3d_agent.config import load_settings
from local_3d_agent.service.app import create_app


class FakeBackend:
    def __init__(self):
        self.calls = []

    def health(self):
        return {"status": "ok", "cuda": True, "gpu": "RTX 5880 Ada", "models": {"hunyuan3d_shape": "ready", "hunyuan3d_paint": "ready", "hunyuandit": "ready"}}

    def generate_image(self, request, job_dir):
        self.calls.append(("image", request.seed, request.shape_steps))
        output = job_dir / "model.glb"; output.write_bytes(b"glb"); return output

    def generate_text(self, request, job_dir):
        self.calls.append(("text", request.prompt, request.seed))
        output = job_dir / "model.glb"; output.write_bytes(b"glb"); return output

    def generate_texture(self, request, job_dir):
        self.calls.append(("texture", request.face_count))
        output = job_dir / "model.glb"; output.write_bytes(b"glb"); return output


def settings_for(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    return load_settings(root / "config/default.yaml", {"LOCAL_3D_PATHS__ASSETS": str(tmp_path)})


def test_health_does_not_load_models(tmp_path: Path) -> None:
    backend = FakeBackend()
    response = TestClient(create_app(settings_for(tmp_path), backend=backend)).get("/health")
    assert response.status_code == 200
    assert response.json()["gpu"] == "RTX 5880 Ada"
    assert backend.calls == []


def test_generate_text_returns_job_output_with_defaults(tmp_path: Path) -> None:
    backend = FakeBackend(); client = TestClient(create_app(settings_for(tmp_path), backend=backend))
    response = client.post("/generate/text", json={"prompt": "industrial robot"})
    assert response.status_code == 200
    assert response.json()["type"] == "glb"
    assert Path(response.json()["file"]).is_file()
    assert backend.calls == [("text", "industrial robot", 12345)]


def test_generate_image_validates_and_calls_backend(tmp_path: Path) -> None:
    image = tmp_path / "input.png"; Image.new("RGB", (8, 8)).save(image)
    backend = FakeBackend(); client = TestClient(create_app(settings_for(tmp_path), backend=backend))
    response = client.post("/generate/image", json={"image": str(image), "seed": 3, "shape_steps": 12})
    assert response.status_code == 200
    assert backend.calls == [("image", 3, 12)]


def test_generate_text_rejects_blank_prompt(tmp_path: Path) -> None:
    response = TestClient(create_app(settings_for(tmp_path), backend=FakeBackend())).post("/generate/text", json={"prompt": "  "})
    assert response.status_code == 422

