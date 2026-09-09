import json
from pathlib import Path

import pytest

from local_3d_agent.model_resolver import (
    ModelSpec,
    ModelStatus,
    load_manifest,
    resolve_model,
    validate_model,
)


def test_load_manifest_returns_named_model_specs(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "models:\n  demo:\n    local_path: /models/demo\n    remote_repo: owner/demo\n    required_files: [config.json]\n",
        encoding="utf-8",
    )

    specs = load_manifest(manifest)

    assert specs["demo"] == ModelSpec(
        name="demo",
        local_path=Path("/models/demo"),
        remote_repo="owner/demo",
        required_files=("config.json",),
    )


def test_validate_model_reports_ready_for_required_files_and_shards(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    weights = tmp_path / "text_encoder"
    weights.mkdir()
    (weights / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    (weights / "model-00001-of-00002.safetensors").touch()
    (weights / "model-00002-of-00002.safetensors").touch()
    spec = ModelSpec(
        name="demo",
        local_path=tmp_path,
        remote_repo="owner/demo",
        required_files=("model_index.json", "text_encoder/model.safetensors.index.json"),
        required_directories=("text_encoder",),
    )

    result = validate_model(spec, tmp_path)

    assert result.status is ModelStatus.READY
    assert result.missing == ()


def test_validate_model_reports_partial_when_index_shard_is_missing(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    weights = tmp_path / "text_encoder"
    weights.mkdir()
    (weights / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "missing.safetensors"}}), encoding="utf-8"
    )
    spec = ModelSpec(
        name="demo",
        local_path=tmp_path,
        remote_repo="owner/demo",
        required_files=("model_index.json", "text_encoder/model.safetensors.index.json"),
        required_directories=("text_encoder",),
    )

    result = validate_model(spec, tmp_path)

    assert result.status is ModelStatus.PARTIAL
    assert "text_encoder/missing.safetensors" in result.missing


def test_resolve_model_downloads_to_fixed_path_then_returns_local(tmp_path: Path) -> None:
    target = tmp_path / "model"
    spec = ModelSpec(
        name="demo",
        local_path=target,
        remote_repo="owner/demo",
        revision="abc123",
        required_files=("model_index.json",),
    )
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model_index.json").write_text("{}", encoding="utf-8")
        return str(target)

    resolved = resolve_model(spec, allow_download=True, downloader=downloader)

    assert calls == [{"repo_id": "owner/demo", "local_dir": target, "revision": "abc123"}]
    assert resolved.local_path == target.resolve()
    assert resolved.downloaded is True
    assert resolved.source == "huggingface"
    assert resolved.validation.status is ModelStatus.READY


def test_resolve_model_refuses_missing_model_when_download_disabled(tmp_path: Path) -> None:
    spec = ModelSpec(
        name="demo", local_path=tmp_path / "missing", remote_repo="owner/demo", required_files=("config.json",)
    )

    with pytest.raises(RuntimeError, match="MISSING"):
        resolve_model(spec, allow_download=False)
