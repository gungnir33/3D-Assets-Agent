from pathlib import Path
import json
from PIL import Image
import trimesh
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

    def resolved_models(self):
        return {
            "hunyuan3d": {
                "path": "/models/h3d",
                "revision": "h3d-rev",
                "downloaded": False,
            }
        }


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
    metadata = json.loads(Path(response.json()["metadata"]).read_text())
    assert metadata["models"]["hunyuan3d"]["path"] == "/models/h3d"
    assert metadata["stages"]["generation"]["duration_seconds"] >= 0
    assert isinstance(metadata["peak_vram_bytes"], int)
    assert metadata["condition_image"].endswith("/condition.png")
    assert Path(response.json()["metadata"]).with_name("generation.log").is_file()


def test_generate_image_validates_and_calls_backend(tmp_path: Path) -> None:
    image = tmp_path / "input.png"; Image.new("RGB", (8, 8)).save(image)
    backend = FakeBackend(); client = TestClient(create_app(settings_for(tmp_path), backend=backend))
    response = client.post("/generate/image", json={"image": str(image), "seed": 3, "shape_steps": 12})
    assert response.status_code == 200
    assert backend.calls == [("image", 3, 12)]


def test_generate_text_honors_obj_output_format(tmp_path: Path) -> None:
    backend = FakeBackend()
    def generate_mesh(request, job):
        output = job / "model.glb"
        trimesh.creation.box().export(output)
        return output
    backend.generate_text = generate_mesh
    response = TestClient(create_app(settings_for(tmp_path), backend=backend)).post(
        "/generate/text", json={"prompt": "robot", "format": "obj", "texture": False}
    )

    assert response.status_code == 200
    assert response.json()["type"] == "obj"
    assert Path(response.json()["file"]).suffix == ".obj"
    assert trimesh.load(response.json()["file"], force="mesh").faces.shape[0] > 0


def test_generate_text_rejects_blank_prompt(tmp_path: Path) -> None:
    response = TestClient(create_app(settings_for(tmp_path), backend=FakeBackend())).post("/generate/text", json={"prompt": "  "})
    assert response.status_code == 422


def test_cuda_oom_returns_structured_memory_details(tmp_path: Path) -> None:
    import torch
    backend = FakeBackend()
    backend.generate_text = lambda request, job: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("full"))
    response = TestClient(create_app(settings_for(tmp_path), backend=backend)).post(
        "/generate/text", json={"prompt": "robot"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "CUDA_OUT_OF_MEMORY"
    assert set(response.json()["error"]["details"]) == {"allocated_bytes", "reserved_bytes", "total_bytes"}
