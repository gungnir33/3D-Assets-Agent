from pathlib import Path
from types import SimpleNamespace
import pytest
from local_3d_agent.config import load_settings
from local_3d_agent.mesh.validate import validate_mesh
from local_3d_agent.service.backend import GenerationBackend

ROOT = Path(__file__).resolve().parents[2]

def backend():
    settings = load_settings(ROOT / "config/default.yaml")
    return GenerationBackend(settings, ROOT / "config/model_manifest.yaml")

@pytest.mark.gpu
def test_image_to_3d_real_gpu(tmp_path: Path):
    image = Path("/home/mcl/models/hunyuan3d/Hunyuan3D-2/assets/demo.png")
    request = SimpleNamespace(image=image, texture=False, seed=12345, shape_steps=50, face_count=40000, format="glb")
    output = backend().generate_image(request, tmp_path)
    assert validate_mesh(output).valid

@pytest.mark.gpu
def test_text_to_3d_real_gpu(tmp_path: Path):
    request = SimpleNamespace(prompt="a futuristic industrial quadruped robot", texture=False, seed=12345,
                              shape_steps=50, face_count=40000, format="glb")
    output = backend().generate_text(request, tmp_path)
    assert (tmp_path / "condition.png").is_file()
    assert validate_mesh(output).valid

@pytest.mark.gpu
def test_texture_real_gpu(tmp_path: Path):
    source = Path("/home/mcl/models/hunyuan3d/Hunyuan3D-2/assets/demo.png")
    shape_request = SimpleNamespace(image=source, texture=False, seed=12345, shape_steps=50, face_count=40000, format="glb")
    shape_job = tmp_path / "shape"
    texture_job = tmp_path / "texture"
    shape_job.mkdir()
    texture_job.mkdir()
    service = backend(); raw = service.generate_image(shape_request, shape_job)
    texture_request = SimpleNamespace(mesh=raw, condition_image=source, texture=True, seed=12345,
                                      shape_steps=50, face_count=40000, format="glb")
    output = service.generate_texture(texture_request, texture_job)
    assert validate_mesh(output, require_texture=True).valid
