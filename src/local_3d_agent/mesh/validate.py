from dataclasses import dataclass
from pathlib import Path
import numpy as np
import trimesh

@dataclass(frozen=True)
class MeshValidation:
    valid: bool
    vertices: int
    faces: int
    textured: bool

def _as_mesh(source: Path | trimesh.Trimesh) -> trimesh.Trimesh:
    if isinstance(source, trimesh.Trimesh):
        return source
    path = Path(source).resolve()
    if not path.is_file():
        raise ValueError(f"mesh file does not exist: {path}")
    if path.suffix.lower() not in {".glb", ".gltf", ".obj", ".ply", ".stl"}:
        raise ValueError(f"unsupported mesh format: {path.suffix}")
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("mesh has no geometry")
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded

def validate_mesh(source: Path | trimesh.Trimesh, *, require_texture: bool = False) -> MeshValidation:
    mesh = _as_mesh(source)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("mesh must contain vertices and faces")
    if not np.isfinite(vertices).all():
        raise ValueError("mesh coordinates must be finite")
    bounds = np.asarray(mesh.bounds)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or not np.any(bounds[1] > bounds[0]):
        raise ValueError("mesh bounding box is invalid")
    visual = mesh.visual
    textured = bool(getattr(visual, "uv", None) is not None and getattr(visual, "material", None) is not None)
    if require_texture and not textured:
        raise ValueError("textured mesh must contain UV and material")
    return MeshValidation(True, len(vertices), len(faces), textured)

