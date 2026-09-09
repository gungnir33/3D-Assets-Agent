# API

The V0.1 service binds only to `127.0.0.1:8080`. `GET /health` checks CUDA, configured paths, and model manifests without loading large models.

`POST /generate/text` accepts `prompt`; `POST /generate/image` accepts an absolute local `image`; `POST /generate/texture` accepts absolute `mesh` and `condition_image`. Common fields are `texture`, `seed`, `format`, `face_count`, and `shape_steps`. Defaults are `true`, `12345`, `glb`, `40000`, and `50`.

Successful generation returns `job_id`, `file`, `type`, and `metadata`. Validation failures use HTTP 422. Runtime errors use `{ "error": { "code", "message", "details" } }`; CUDA OOM uses HTTP 503 and includes allocated, reserved, and total bytes.

Clients must use a timeout of at least 900 seconds. LAN exposure, uploads, authentication, and an asynchronous queue are outside V0.1.

