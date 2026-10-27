import importlib.metadata
import json

from openpi_client.e0 import rollout
from openpi_client.e0.artifacts import write_jsonl
from openpi_client.e0.protocol import episodes
import pytest


class FakeClient:
    def __init__(self, *args):
        self.metadata = {"policy_id": "test-policy", "H": 10, "K": 5}

    def close(self):
        pass


def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(rollout, "Client", FakeClient)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test")
    manifest = tmp_path / "episodes.jsonl"
    write_jsonl(manifest, list(episodes("smoke/retry", ["libero_10"], attempts=1))[:1])
    return manifest, tmp_path / "results"


def test_worker_interruption_consumes_retry(monkeypatch, tmp_path):
    manifest, out = setup(monkeypatch, tmp_path)

    def interrupted(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(rollout, "one_episode", interrupted)
    with pytest.raises(KeyboardInterrupt):
        rollout.run(manifest, out)

    def completed(*args):
        return [], {"success": False, "policy_steps": 0, "status": "policy_failure", "error": "test", "prompt": "test"}

    monkeypatch.setattr(rollout, "one_episode", completed)
    rollout.run(manifest, out)
    result = json.loads(next((out / "attempts").glob("*.json")).read_text())
    assert result["retry"] == 1
    assert len(result["previous_errors"]) == 1
    # A committed attempt must not run again.
    monkeypatch.setattr(rollout, "one_episode", interrupted)
    rollout.run(manifest, out)


def test_infrastructure_retry_never_resets(monkeypatch, tmp_path):
    manifest, out = setup(monkeypatch, tmp_path)
    calls = []

    def broken(*args):
        calls.append(1)
        raise rollout.InfrastructureError("connection lost")

    monkeypatch.setattr(rollout, "one_episode", broken)
    with pytest.raises(rollout.InfrastructureError):
        rollout.run(manifest, out)
    with pytest.raises(rollout.InfrastructureError):
        rollout.run(manifest, out)
    assert len(calls) == 2
