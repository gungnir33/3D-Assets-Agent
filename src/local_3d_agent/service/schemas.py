from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator

class GenerationBase(BaseModel):
    texture: bool = True
    seed: int = Field(12345, ge=0, le=2**63 - 1)
    format: Literal["glb", "obj", "fbx"] = "glb"
    face_count: int = Field(40000, ge=100, le=1_000_000)
    shape_steps: int = Field(50, ge=1, le=200)

class TextRequest(GenerationBase):
    prompt: str = Field(min_length=1, max_length=1000)
    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt must not be blank")
        return value

def _validate_file(path: Path, suffixes: set[str], label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label} must be an existing regular file")
    if path.suffix.lower() not in suffixes:
        raise ValueError(f"unsupported {label} format")
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError(f"{label} exceeds 100 MiB")
    return path

class ImageRequest(GenerationBase):
    image: Path
    @field_validator("image")
    @classmethod
    def valid_image(cls, value: Path) -> Path:
        return _validate_file(value, {".png", ".jpg", ".jpeg", ".webp"}, "image")

class TextureRequest(GenerationBase):
    mesh: Path
    condition_image: Path
    @field_validator("mesh")
    @classmethod
    def valid_mesh(cls, value: Path) -> Path:
        return _validate_file(value, {".glb", ".gltf", ".obj", ".ply", ".stl"}, "mesh")
    @field_validator("condition_image")
    @classmethod
    def valid_condition(cls, value: Path) -> Path:
        return _validate_file(value, {".png", ".jpg", ".jpeg", ".webp"}, "condition image")

class GenerationResponse(BaseModel):
    job_id: str
    file: str
    type: str
    metadata: str

