from pathlib import Path

import pytest

from local_3d_agent.config import load_settings


def test_load_settings_applies_yaml_defaults_and_environment_overrides(tmp_path: Path) -> None:
    config = tmp_path / "default.yaml"
    config.write_text(
        """
server: {host: 127.0.0.1, port: 8080}
paths:
  hunyuan3d_source: /source
  hunyuan3d_model: /models/h3d
  hunyuandit_model: /models/dit
  rembg_model_dir: /models/rembg
  assets: /assets
models: {allow_download: true}
runtime: {device: cuda, dtype: float16}
generation: {default_seed: 12345, default_format: glb, default_face_count: 40000}
""",
        encoding="utf-8",
    )

    settings = load_settings(
        config,
        {
            "LOCAL_3D_SERVER__PORT": "9090",
            "LOCAL_3D_MODELS__ALLOW_DOWNLOAD": "false",
            "HUNYUAN3D_SOURCE_ROOT": "/override/source",
            "U2NET_HOME": "/override/rembg",
        },
    )

    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 9090
    assert settings.models.allow_download is False
    assert settings.paths.hunyuan3d_source == Path("/override/source")
    assert settings.paths.rembg_model_dir == Path("/override/rembg")
    assert settings.generation.default_face_count == 40000


def test_load_settings_rejects_relative_model_paths(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        """
paths:
  hunyuan3d_source: relative/source
  hunyuan3d_model: /models/h3d
  hunyuandit_model: /models/dit
  rembg_model_dir: /models/rembg
  assets: /assets
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absolute"):
        load_settings(config, {})

