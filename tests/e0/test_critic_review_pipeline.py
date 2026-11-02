"""Exercise the full offline report on a tiny synthetic training/review pair."""

import json
from pathlib import Path

import numpy as np
import pytest
from openpi_client.e0.artifacts import content_id, write_json

from openpi.e0.critic_report import read_thresholds, report
from openpi.e0.reward_study import METHODS, MODES, fit_scales, save_labels


def test_review_report_and_role_binding(tmp_path):
    data = {}
    for split, source, count in (("pool", "rollout", 1), ("review", "review", 2)):
        rows = [
            dict(episode_uid=f"{source}-{i}", task_index=0, episode_index=i, length=11, source=source, success=i == 1)
            for i in range(count)
        ]
        snapshot = dict(episodes=rows, long_task_indices=[0])
        snapshot["dataset_id"] = content_id(snapshot)
        write_json(tmp_path / split / "e0-manifest.json", snapshot)
        data["train" if split == "pool" else split] = snapshot
    for ep in data["review"]["episodes"]:
        uid = ep["episode_uid"]
        image = np.zeros((11, 16, 16, 3), np.uint8)
        np.savez_compressed(tmp_path / (uid + ".npz"), image=image, wrist_image=image)
        write_json(
            tmp_path / (uid + ".json"),
            dict(
                episode_uid=uid,
                policy_id="fixture",
                decisions=[
                    dict(start_step=0, end_step=5),
                    dict(start_step=5, end_step=10),
                    dict(start_step=10, end_step=11),
                ],
            ),
        )
        write_json(
            tmp_path / "review-evaluation/results/attempts" / (uid + ".json"),
            dict(
                episode_uid=uid,
                task_id=0,
                success=ep["success"],
                policy_steps=11,
                prompt="fixture",
                trajectory=str(tmp_path / (uid + ".npz")),
                decisions_path=str(tmp_path / (uid + ".json")),
                policy_id="fixture",
            ),
        )
    write_json(tmp_path / "target-differences.json", {})
    for method in METHODS:
        labels = tmp_path / ("labels-" + method)
        rec = save_labels(labels, method, fit_scales(data["train"]), data)
        for mode in MODES:
            base = tmp_path / (method + "-" + mode)
            spec = dict(
                reward_protocol=str(labels / "protocol.json"),
                reward_protocol_id=rec["protocol_id"],
                dataset_id=data["train"]["dataset_id"],
                manifest_id=method + "-" + mode,
            )
            write_json(base / "training.json", spec)
            write_json(
                base / "selection.json", dict(checkpoint="fixture", updates=10000, validation_dataset_id="selection")
            )
            for role in data:
                (base / role).mkdir()
                rows = []
                for ep in data[role]["episodes"]:
                    n = ep["length"]
                    uid = ep["episode_uid"]
                    np.save(base / role / (uid + ".npy"), np.linspace(-0.85, -0.05, n))
                    np.save(base / role / (uid + "-ce.npy"), np.ones(n))
                    rows.append(dict(episode_uid=uid, length=n, path=uid + ".npy", ce_path=uid + "-ce.npy"))
                write_json(
                    base / role / "predictions.json",
                    dict(
                        reward_protocol_id=rec["protocol_id"],
                        role=role,
                        checkpoint="fixture",
                        dataset_id=data[role]["dataset_id"],
                        manifest_id=spec["manifest_id"],
                        prediction_units="reward",
                        episodes=rows,
                    ),
                )
    dest = tmp_path / "report"
    report(tmp_path, dest)
    summary = json.loads((dest / "study.json").read_text())
    assert len(summary["models"]) == 6 and len(summary["pairs"]) == 9
    assert len(list(dest.glob("*.mp4"))) == 2
    assert (dest / "index.html").exists() and (dest / "disagreements.html").exists()
    with pytest.raises(ValueError, match="threshold binding"):
        read_thresholds(
            dest / base.name / "thresholds.json", spec, "different-checkpoint", data["train"]["dataset_id"]
        )
    trace = (dest / "trace-000.html").read_text()
    assert "__PAYLOAD__" not in trace and "window_reward" in trace
    path = base / "review/predictions.json"
    pred = json.loads(path.read_text())
    pred["reward_protocol_id"] = "wrong"
    write_json(path, pred)
    with pytest.raises(ValueError, match="wrong predictions"):
        report(tmp_path, tmp_path / "bad")
