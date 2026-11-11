import json
from pathlib import Path

import pytest
from openpi_client.e0.artifacts import write_json, write_jsonl
from openpi_client.e0.protocol import episodes

from openpi.e0 import actor_confirmation as confirmation
from openpi.e0.evaluation import validate_episode_manifest


@pytest.mark.parametrize("purpose,groups", [("review", [0]), ("review", [0, 1, 2]), ("final", [0, 1, 2]), ("development", [0])])
def test_complete_evaluation_groups(purpose, groups):
    rows = list(episodes("test/confirm", ["libero_10"], groups=groups, purpose=purpose, training_seed=42))
    validate_episode_manifest(dict(seed=42, smoke=False, stage="r1"), rows)


@pytest.mark.parametrize("defect", ["missing_episode", "missing_group", "duplicate", "changed_seed"])
def test_review_grid_still_rejects_invalid_manifests(defect):
    rows = list(episodes("test/confirm", ["libero_10"], groups=[0, 1, 2], purpose="review", training_seed=42))
    if defect == "missing_episode":
        rows.pop()
    elif defect == "missing_group":
        rows = [r for r in rows if r["group"] != 1]
    elif defect == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[0]["sampling_seed"] += 1
    with pytest.raises(ValueError):
        validate_episode_manifest(dict(seed=42, smoke=False, stage="r1"), rows)


@pytest.fixture
def completed_study(tmp_path, monkeypatch):
    write_json(tmp_path / "execution.json", dict(runtime_id="same-environment"))
    write_jsonl(tmp_path / "confirmation-episodes.jsonl", episodes("test/confirm", ["libero_10"], groups=[0, 1, 2], purpose="review", training_seed=42))
    for name in ("B", "C", "SFT"):
        root = tmp_path / name
        spec = dict(run_dir=str(root / "run"), num_train_steps=60000, seed=42, stage="r1", branch="base_reset", manifest_id=name)
        write_json(root / "training.json", spec)
        write_json(root / "run/resolved-manifest.json", spec)
        write_json(root / "run/status.json", dict(complete=True, completed_updates=60000))
        write_json(root / "checkpoint/e0-checkpoint.json", dict(manifest_id=name, parameter_kind="EMA"))
        selection = dict(updates=40000 if name == "SFT" else 30000, checkpoint=str(root / "checkpoint"))
        write_json(root / "selection.json", selection)
        write_json(root / "evaluations.json", selection)
    write_json(tmp_path / "comparison.json", dict(sft_selection=json.loads((tmp_path / "SFT/selection.json").read_text())))
    monkeypatch.setattr(confirmation, "read_manifest", lambda p: json.loads(Path(p).read_text()))
    monkeypatch.setattr(confirmation, "SFT", tmp_path / "SFT/training.json")
    monkeypatch.setattr(confirmation, "actor", lambda manifest, evaluations, output: write_json(output, json.loads(Path(evaluations).read_text())))
    return tmp_path


def test_confirmation_uses_completed_selected_checkpoints(completed_study):
    selected, files = confirmation.selected_inputs(completed_study)
    assert list(selected) == ["B", "C", "SFT"]
    assert all(Path(c).is_dir() for _, c in selected.values())
    assert all(p.exists() for p in files)


@pytest.mark.parametrize("defect", ["unfinished", "changed_selection", "wrong_checkpoint"])
def test_confirmation_rejects_unsealed_inputs(completed_study, defect):
    p = completed_study
    if defect == "unfinished":
        write_json(p / "C/run/status.json", dict(complete=False, completed_updates=45000))
    elif defect == "changed_selection":
        write_json(p / "C/selection.json", dict(updates=45000, checkpoint="different"))
    else:
        write_json(p / "C/checkpoint/e0-checkpoint.json", dict(manifest_id="wrong", parameter_kind="EMA"))
    with pytest.raises(ValueError):
        confirmation.selected_inputs(p)
