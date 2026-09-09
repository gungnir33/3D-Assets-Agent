from pathlib import Path
import numpy as np
import pytest
import trimesh

from local_3d_agent.mesh.preprocess import preprocess_mesh
from local_3d_agent.mesh.validate import validate_mesh


def test_validate_mesh_accepts_reloadable_nonempty_mesh(tmp_path: Path) -> None:
    path = tmp_path / "box.glb"
    trimesh.creation.box().export(path)

    result = validate_mesh(path)

    assert result.vertices == 8
    assert result.faces == 12
    assert result.valid is True


def test_validate_mesh_rejects_nonfinite_coordinates(tmp_path: Path) -> None:
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [np.nan, 1, 0]], faces=[[0, 1, 2]], process=False)

    with pytest.raises(ValueError, match="finite"):
        validate_mesh(mesh)


def test_preprocess_mesh_calls_processors_in_required_order() -> None:
    events = []
    mesh = object()
    floater = lambda value: events.append("floater") or value
    degenerate = lambda value: events.append("degenerate") or value
    reducer = lambda value, max_facenum: events.append(f"reduce:{max_facenum}") or value

    result = preprocess_mesh(mesh, 40000, floater=floater, degenerate=degenerate, reducer=reducer)

    assert result is mesh
    assert events == ["floater", "degenerate", "reduce:40000"]

