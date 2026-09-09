"""Validated application configuration."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Mapping
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

class ServerSettings(StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(8080, ge=1, le=65535)

class PathSettings(StrictModel):
    hunyuan3d_source: Path
    hunyuan3d_model: Path
    hunyuandit_model: Path
    rembg_model_dir: Path
    assets: Path
    @field_validator("*", mode="after")
    @classmethod
    def absolute_paths_only(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("configured paths must be absolute")
        return value

class ModelSettings(StrictModel):
    allow_download: bool = True

class RuntimeSettings(StrictModel):
    device: str = "cuda"
    dtype: str = "float16"

class GenerationSettings(StrictModel):
    default_seed: int = Field(12345, ge=0, le=2**63 - 1)
    default_format: str = "glb"
    default_face_count: int = Field(40000, ge=100, le=1_000_000)
    default_shape_steps: int = Field(50, ge=1, le=200)

class Settings(StrictModel):
    server: ServerSettings = ServerSettings()
    paths: PathSettings
    models: ModelSettings = ModelSettings()
    runtime: RuntimeSettings = RuntimeSettings()
    generation: GenerationSettings = GenerationSettings()

def load_settings(config_path: Path, environ: Mapping[str, str] | None = None) -> Settings:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    env = dict(os.environ if environ is None else environ)
    for key, value in env.items():
        if key.startswith("LOCAL_3D_") and "__" in key:
            section, field = key.removeprefix("LOCAL_3D_").lower().split("__", 1)
            if section in data and isinstance(data[section], dict):
                data[section][field] = yaml.safe_load(value)
    if env.get("HUNYUAN3D_SOURCE_ROOT"):
        data.setdefault("paths", {})["hunyuan3d_source"] = env["HUNYUAN3D_SOURCE_ROOT"]
    if env.get("U2NET_HOME"):
        data.setdefault("paths", {})["rembg_model_dir"] = env["U2NET_HOME"]
    return Settings.model_validate(data)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

