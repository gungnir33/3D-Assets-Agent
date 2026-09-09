from __future__ import annotations
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

def create_job(assets: Path, *, now: Callable[[], str] | None = None,
               short_id: Callable[[], str] | None = None) -> Path:
    stamp = (now or (lambda: datetime.now().strftime("%Y%m%d_%H%M%S")))()
    suffix = (short_id or (lambda: uuid.uuid4().hex[:8]))()
    job = assets / f"{stamp}_{suffix}"
    job.mkdir(parents=True, exist_ok=False)
    return job

class JobMetadata:
    def __init__(self, job_dir: Path, *, seed: int, prompt: str | None,
                 shape_steps: int, face_count: int, device: str, dtype: str):
        self.job_dir = job_dir
        self.started_monotonic = time.monotonic()
        self.data = {
            "status": "RUNNING", "seed": seed, "prompt": prompt,
            "shape_steps": shape_steps, "face_count": face_count,
            "device": device, "dtype": dtype,
            "generation_start_time": datetime.now(timezone.utc).isoformat(),
            "models": {}, "stages": {}, "peak_vram_bytes": 0,
        }
        self._write()

    def _write(self) -> None:
        target = self.job_dir / "metadata.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)

    def record_model(self, name: str, path: str, revision: str | None, *, downloaded: bool) -> None:
        self.data["models"][name] = {"path": path, "revision": revision, "downloaded": downloaded}
        self._write()

    def record_stage(self, name: str, duration: float, *, peak_vram: int = 0) -> None:
        self.data["stages"][name] = {"duration_seconds": duration}
        self.data["peak_vram_bytes"] = max(self.data["peak_vram_bytes"], peak_vram)
        self._write()

    def finish(self, output: Path) -> None:
        self.data.update(status="SUCCEEDED", output=str(output),
                         generation_duration_seconds=time.monotonic() - self.started_monotonic)
        self._write()

    def fail(self, code: str, message: str) -> None:
        self.data.update(status="FAILED", error={"code": code, "message": message},
                         generation_duration_seconds=time.monotonic() - self.started_monotonic)
        self._write()

