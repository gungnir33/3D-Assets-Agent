from pathlib import Path
import shutil
import subprocess
from .validate import validate_mesh

def export_mesh(mesh, output: Path, format: str = "glb") -> Path:
    fmt = format.lower()
    if fmt not in {"glb", "obj", "fbx"}:
        raise ValueError(f"unsupported output format: {format}")
    output = output.with_suffix(f".{fmt}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "fbx":
        blender = shutil.which("blender")
        if blender is None:
            raise RuntimeError("FBX export requires Blender")
        raise RuntimeError("FBX conversion requires a source file and is unavailable for in-memory meshes")
    mesh.export(output)
    validate_mesh(output)
    return output
