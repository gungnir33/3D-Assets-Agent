"""Central local-first model validation and resolution."""
from __future__ import annotations
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable
import yaml

class ModelStatus(str, Enum):
    READY = "READY"
    MISSING = "MISSING"
    PARTIAL = "PARTIAL"

@dataclass(frozen=True)
class ModelSpec:
    name: str
    local_path: Path
    remote_repo: str | None
    revision: str | None = None
    required_files: tuple[str, ...] = ()
    required_directories: tuple[str, ...] = ()
    optional_files: tuple[str, ...] = ()

@dataclass(frozen=True)
class ModelValidation:
    status: ModelStatus
    missing: tuple[str, ...]

@dataclass(frozen=True)
class ResolvedModel:
    local_path: Path
    source: str
    downloaded: bool
    revision: str | None
    validation: ModelValidation

def load_manifest(path: Path) -> dict[str, ModelSpec]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_models = payload.get("models")
    if not isinstance(raw_models, dict):
        raise ValueError("manifest must contain a models mapping")
    specs: dict[str, ModelSpec] = {}
    for name, raw in raw_models.items():
        if not isinstance(raw, dict) or "local_path" not in raw:
            raise ValueError(f"invalid model manifest entry: {name}")
        specs[name] = ModelSpec(
            name=name,
            local_path=Path(raw["local_path"]),
            remote_repo=raw.get("remote_repo"),
            revision=raw.get("revision"),
            required_files=tuple(raw.get("required_files", ())),
            required_directories=tuple(raw.get("required_directories", ())),
            optional_files=tuple(raw.get("optional_files", ())),
        )
    return specs

def _indexed_shards(index_path: Path) -> set[str]:
    if not index_path.name.endswith(".safetensors.index.json"):
        return set()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return set(payload.get("weight_map", {}).values())

def validate_model(spec: ModelSpec, local_path: Path | None = None) -> ModelValidation:
    root = Path(local_path or spec.local_path)
    if not root.is_dir():
        return ModelValidation(ModelStatus.MISSING, (str(root),))
    missing: list[str] = []
    for relative in spec.required_directories:
        if not (root / relative).is_dir():
            missing.append(relative)
    for relative in spec.required_files:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        for shard in _indexed_shards(path):
            shard_relative = str(Path(relative).parent / shard)
            if not (root / shard_relative).is_file():
                missing.append(shard_relative)
    status = ModelStatus.READY if not missing else ModelStatus.PARTIAL
    return ModelValidation(status, tuple(sorted(set(missing))))

def _hub_download(**kwargs: object) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(**kwargs)

def resolve_model(spec: ModelSpec, *, allow_download: bool = True,
                  downloader: Callable[..., str] = _hub_download) -> ResolvedModel:
    target = spec.local_path.expanduser().resolve()
    validation = validate_model(spec, target)
    if validation.status is ModelStatus.READY:
        return ResolvedModel(target, "local", False, spec.revision, validation)
    if not allow_download:
        raise RuntimeError(f"{spec.name} is {validation.status.value}: {', '.join(validation.missing)}")
    if not spec.remote_repo:
        raise RuntimeError(f"{spec.name} is {validation.status.value} and has no download source")
    target.mkdir(parents=True, exist_ok=True)
    downloader(repo_id=spec.remote_repo, local_dir=target, revision=spec.revision)
    validation = validate_model(spec, target)
    if validation.status is not ModelStatus.READY:
        raise RuntimeError(f"{spec.name} remains {validation.status.value} after download: {', '.join(validation.missing)}")
    return ResolvedModel(target, "huggingface", True, spec.revision, validation)
