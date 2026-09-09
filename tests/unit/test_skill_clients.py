import importlib.util
import json
from pathlib import Path

SKILL_SCRIPTS = Path(__file__).resolve().parents[2] / "skill/local-3d-agent/scripts"

def load_client():
    spec = importlib.util.spec_from_file_location("skill_client", SKILL_SCRIPTS / "client.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def test_server_down_has_stable_error_and_start_instruction(monkeypatch, capsys) -> None:
    client = load_client()
    monkeypatch.setattr(client.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(client.httpx.ConnectError("down")))
    assert client.check_health() is False
    output = capsys.readouterr().out
    assert "LOCAL_3D_SERVER_NOT_RUNNING" in output
    assert "/home/mcl/workspace/3D-Assets-Agent/scripts/start_server.sh" in output

def test_post_generation_uses_long_timeout_and_prints_result(monkeypatch, capsys) -> None:
    client = load_client(); calls = []
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"file": "/assets/model.glb", "type": "glb"}
    monkeypatch.setattr(client, "check_health", lambda: True)
    monkeypatch.setattr(client.httpx, "post", lambda url, json, timeout: calls.append((url, json, timeout)) or Response())
    assert client.post_generation("text", {"prompt": "robot"}) == 0
    assert calls[0][2] >= 900
    assert json.loads(capsys.readouterr().out)["file"] == "/assets/model.glb"

