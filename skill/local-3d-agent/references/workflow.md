# Local 3D workflow reference

The API base URL is `http://127.0.0.1:8080`. Generation defaults are textured GLB, seed 12345, 40,000 faces, and 50 shape steps. Use `--no-texture` for geometry-only output. FBX works only when Blender is installed.

The service stores every request under `/home/mcl/workspace/3D-Assets-Agent/assets/<job-id>/` with `metadata.json`, intermediate files, logs when available, and final `model.glb`.

Errors:

- `LOCAL_3D_SERVER_NOT_RUNNING`: start the foreground service with `/home/mcl/workspace/3D-Assets-Agent/scripts/start_server.sh`.
- `MODEL_MISSING` or `MODEL_PARTIAL`: run `python scripts/check_models.py`; use `download_models.py` only when the user permits missing weights to be downloaded.
- `GENERATION_FAILED`: inspect the returned metadata and generation log. Do not kill other GPU processes automatically.

