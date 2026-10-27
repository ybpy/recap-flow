import json

from openpi_client.e0.artifacts import content_id
from openpi_client.e0.artifacts import write_json
import pytest

from openpi.e0.config import read_manifest
from openpi.e0.jobs import freeze


def critic_manifest(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    dataset = {"episodes": [], "long_task_indices": list(range(10))}
    dataset["dataset_id"] = content_id(dataset)
    write_json(data / "e0-manifest.json", dataset)
    spec = {
        "id": "unit-critic",
        "kind": "critic",
        "stage": "r1",
        "seed": 42,
        "branch": "shared",
        "run_dir": str(tmp_path / "run"),
        "data_dir": str(data),
        "dataset_manifest": str(data / "e0-manifest.json"),
        "batch_size": 4,
        "num_train_steps": 20,
        "warmup_steps": 2,
        "fsdp_devices": 1,
        "num_workers": 0,
        "checkpoint_interval": 10,
        "checkpoint_updates": [20],
        "smoke": True,
    }
    path = tmp_path / "job.json"
    write_json(path, spec)
    return path


def test_sealed_job_cannot_change_parameters(tmp_path):
    path = critic_manifest(tmp_path)
    sealed = read_manifest(path)
    write_json(path, sealed)
    assert read_manifest(path) == sealed
    sealed["batch_size"] = 8
    write_json(path, sealed)
    with pytest.raises(ValueError, match="sealed task manifest checksum"):
        read_manifest(path)


def test_sealed_job_cannot_switch_dataset_version(tmp_path):
    path = critic_manifest(tmp_path)
    sealed = read_manifest(path)
    write_json(path, sealed)
    dataset = {"episodes": [], "long_task_indices": list(range(10)), "different_snapshot": True}
    dataset["dataset_id"] = content_id(dataset)
    write_json(sealed["dataset_manifest"], dataset)
    with pytest.raises(ValueError, match="dataset version differs"):
        read_manifest(path)


def test_round_specific_critic_budgets(tmp_path):
    evidence = tmp_path / "pilot-evidence.json"
    write_json(evidence, {"unit_fixture": True})
    calibration = {
        "actor_updates": dict.fromkeys(("sft", "r1", "r2"), 30000),
        "actor_candidates": {stage: [24000, 26000, 28000, 30000] for stage in ("sft", "r1", "r2")},
        "critic_updates": {"r1": 10000, "r2": 20000},
        "critic_candidates": {"r1": [8000, 10000], "r2": [16000, 20000]},
        "critic_freeze_mode": "none",
        "evidence": [str(evidence)],
    }
    path, output = tmp_path / "calibration.json", tmp_path / "frozen.json"
    write_json(path, calibration)
    freeze(path, output)
    assert json.loads(output.read_text())["critic_updates"] == {"r1": 10000, "r2": 20000}
    calibration["critic_candidates"]["r2"] = [21000]
    write_json(path, calibration)
    with pytest.raises(ValueError, match="invalid critic budget/candidates for r2"):
        freeze(path, output)
