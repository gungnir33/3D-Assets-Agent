import json
from pathlib import Path

from local_3d_agent.metadata import JobMetadata, create_job


def test_metadata_writes_required_generation_fields(tmp_path: Path) -> None:
    job = create_job(tmp_path, now=lambda: "20260909_120000", short_id=lambda: "abc123")
    metadata = JobMetadata(job, seed=7, prompt="robot", shape_steps=20, face_count=40000,
                           device="cuda", dtype="float16")
    metadata.record_model("shape", "/models/h3d", "rev1", downloaded=False)
    metadata.record_stage("shape", 1.25, peak_vram=1024)
    metadata.finish(job / "model.glb")

    payload = json.loads((job / "metadata.json").read_text())
    assert job.name == "20260909_120000_abc123"
    assert payload["seed"] == 7
    assert payload["models"]["shape"]["revision"] == "rev1"
    assert payload["stages"]["shape"]["duration_seconds"] == 1.25
    assert payload["peak_vram_bytes"] == 1024
    assert payload["status"] == "SUCCEEDED"

