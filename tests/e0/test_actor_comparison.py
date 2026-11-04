import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from openpi_client.e0.artifacts import content_id, sha256_file, write_json, write_jsonl

from openpi.e0.actor_labels import relabel_study, validate_label_binding
from openpi.e0.actor_report import interval
from openpi.e0.critic_report import parts
from openpi.e0.reward_study import fit_scales, save_labels


@pytest.fixture
def study_inputs(tmp_path):
    source, pred, ckpt = [tmp_path / k for k in ("source", "predictions", "checkpoint")]
    episodes = []
    schema = pa.schema(
        [
            pa.field("image", pa.binary()),
            pa.field("state", pa.list_(pa.float32(), 2)),
            pa.field("actions", pa.list_(pa.float32(), 2)),
            pa.field("reward", pa.float32()),
            pa.field("reward_label", pa.list_(pa.float32(), 1)),
            pa.field("value_label", pa.list_(pa.float32(), 1)),
            pa.field("adv_ind", pa.string()),
            pa.field("intervention", pa.int64()),
        ],
        metadata={b"fixture": b"preserve-image-schema"},
    )
    for i, kind in enumerate(("demo", "rollout")):
        path = source / f"episode-{i}.parquet"
        path.parent.mkdir(exist_ok=True)
        table = pa.Table.from_pylist(
            [
                dict(
                    image=b"imagebytes",
                    state=[t, 1],
                    actions=[t, 2],
                    reward=0.0,
                    reward_label=[-0.01],
                    value_label=[-1.0],
                    adv_ind="positive" if kind == "demo" else "none",
                    intervention=int(kind == "demo"),
                )
                for t in range(60)
            ],
            schema=schema,
        )
        pq.write_table(table, path)
        episodes.append(
            dict(
                episode_uid=kind,
                episode_index=i,
                task_index=0,
                length=60,
                source=kind,
                success=False,
                path=path.name,
                sha256=sha256_file(path),
            )
        )
    data = dict(episodes=episodes, long_task_indices=[0])
    data["dataset_id"] = content_id(data)
    write_json(source / "e0-manifest.json", data)
    write_jsonl(source / "meta/episodes_stats.jsonl", [dict(episode_index=i, stats={}) for i in range(2)])
    labels = tmp_path / "labels"
    protocol = save_labels(labels, "global_pool_max_clip", fit_scales(data), {"train": data})
    spec = dict(
        reward_protocol=str(labels / "protocol.json"),
        reward_protocol_id=protocol["protocol_id"],
        dataset_id=data["dataset_id"],
    )
    spec["manifest_id"] = content_id(spec)
    manifest = tmp_path / "critic.json"
    write_json(manifest, spec)
    write_json(ckpt / "e0-checkpoint.json", dict(manifest_id=spec["manifest_id"], parameter_kind="EMA"))
    pred.mkdir()
    values = np.full(60, -0.75, np.float32)
    np.save(pred / "rollout.npy", values)
    write_json(
        pred / "predictions.json",
        dict(
            dataset_id=data["dataset_id"],
            manifest_id=spec["manifest_id"],
            reward_protocol_id=protocol["protocol_id"],
            checkpoint=str(ckpt),
            role="train",
            prediction_units="reward",
            episodes=[dict(episode_uid="rollout", length=60, path="rollout.npy")],
        ),
    )
    targets = np.load(labels / "train.npy")
    a = parts(targets[60:, 0], values)[2]
    thresholds = dict(
        manifest_id=spec["manifest_id"],
        reward_protocol_id=spec["reward_protocol_id"],
        checkpoint=str(ckpt),
        train_dataset_id=data["dataset_id"],
        thresholds={"0": float(np.percentile(a, 70))},
    )
    thresholds["threshold_id"] = content_id(thresholds)
    path = tmp_path / "thresholds.json"
    write_json(path, thresholds)
    return source, pred, manifest, path, targets


def test_actor_snapshot_uses_bound_reward_and_preserves_inputs(tmp_path, study_inputs):
    source, pred, manifest, thresholds, targets = study_inputs
    output = tmp_path / "labeled"
    relabel_study(source, pred, output, manifest, thresholds)
    metadata = json.loads((output / "e0-manifest.json").read_text())
    validate_label_binding(output, metadata)
    for i in range(2):
        old = pq.read_table(source / f"episode-{i}.parquet")
        new = pq.read_table(output / f"episode-{i}.parquet")
        assert old.drop(["reward_label", "value_label", "adv_ind"]).equals(
            new.drop(["reward_label", "value_label", "adv_ind"]), check_metadata=True
        )
        np.testing.assert_array_equal(
            np.array(new["value_label"].to_pylist()).ravel(), targets[i * 60 : (i + 1) * 60, 1]
        )
    assert pq.read_table(output / "episode-0.parquet")["adv_ind"].to_pylist() == ["positive"] * 60
    with np.load(output / "advantage-diagnostics.npz") as a:
        assert a["expert"][:60].all() and not a["expert"][60:].any()
        np.testing.assert_array_equal(a["positive"][60:], a["advantage"][60:] >= a["threshold"][60:])
        np.testing.assert_allclose(a["advantage"][-1], -0.5 - a["value_pred"][-1])
    wrong = json.loads((output / "thresholds.json").read_text())
    wrong["checkpoint"] = "different-checkpoint"
    write_json(output / "thresholds.json", wrong)
    with pytest.raises(ValueError, match="binding differs"):
        validate_label_binding(output, metadata)


@pytest.mark.parametrize("kind", ["role", "protocol", "checkpoint", "missing", "threshold"])
def test_actor_labels_reject_mixed_inputs(tmp_path, study_inputs, kind):
    source, pred, manifest, thresholds, _ = study_inputs
    path = pred / "predictions.json"
    data = json.loads(path.read_text())
    if kind == "threshold":
        t = json.loads(thresholds.read_text())
        t["thresholds"]["0"] += 0.01
        t.pop("threshold_id")
        t["threshold_id"] = content_id(t)
        write_json(thresholds, t)
    else:
        if kind == "role":
            data["role"] = "review"
        if kind == "protocol":
            data["reward_protocol_id"] = "wrong"
        if kind == "checkpoint":
            data["checkpoint"] = str(tmp_path / "other")
            write_json(tmp_path / "other/e0-checkpoint.json", dict(manifest_id="wrong", parameter_kind="EMA"))
        if kind == "missing":
            data["episodes"] = []
        write_json(path, data)
    with pytest.raises(ValueError):
        relabel_study(source, pred, tmp_path / "bad", manifest, thresholds)


def test_actor_bootstrap_keeps_eval_seeds_inside_initial_state():
    draws = np.random.default_rng(42).integers(0, 50, (10, 10000, 50))
    data = np.zeros((10, 50, 3))
    data[:, :, 0] = 1
    result = interval(data, draws)
    assert result["mean"] == pytest.approx(1 / 3)
    np.testing.assert_allclose(result["ci95"], [1 / 3, 1 / 3])
    result = interval(data - data, draws)
    assert result["ci95"] == [0, 0]


def test_complete_actor_report_selects_paired_winner(tmp_path, monkeypatch):
    from openpi_client.e0.protocol import episodes
    from openpi.e0 import actor_report

    expected = list(episodes("fixture", ["libero_10"], groups=[0, 1, 2], purpose="review"))
    write_jsonl(tmp_path / "confirmation-episodes.jsonl", expected)
    for name, cutoff in (("B", 48), ("C", 40), ("SFT", 44)):
        ckpt = tmp_path / name / "checkpoint"
        metadata = dict(name=name, parameter_kind="EMA")
        write_json(ckpt / "e0-checkpoint.json", metadata)
        selection = dict(
            updates=60000,
            checkpoint=str(ckpt),
            candidates=[dict(updates=step, macro_success=cutoff / 50) for step in (30000, 45000, 60000)],
        )
        if name == "SFT":
            write_json(tmp_path / "comparison.json", dict(sft_selection=selection))
        else:
            write_json(tmp_path / name / "selection.json", selection)
            write_json(tmp_path / name / "training.json", dict(run_dir=str(tmp_path / name / "run")))
        results = tmp_path / name / "confirmation/results"
        write_json(results / "protocol.json", dict(policy_id=content_id(metadata)))
        for i, row in enumerate(expected):
            write_json(
                results / "attempts" / f"{i}.json",
                dict(
                    row,
                    success=row["initial_state_id"] < cutoff,
                    status="ok",
                    policy_steps=1,
                    policy_id=content_id(metadata),
                ),
            )
    original = actor_report.read_jsonl
    metrics = [
        dict(
            updates=i,
            rng_key=[i, 42],
            loss=1.0,
            grad_norm=1.0,
            exposure=dict(
                task_frame_counts=[1] * 40,
                rollout_frames=1,
                positive_frames=1,
                negative_frames=0,
                positive_rollout_frames=1,
                observation_action_sha256="same",
            ),
        )
        for i in range(1, 60001)
    ]
    monkeypatch.setattr(
        actor_report, "read_jsonl", lambda p: metrics if Path(p).name == "metrics.jsonl" else original(p)
    )
    actor_report.report(tmp_path, tmp_path / "report")
    output = json.loads((tmp_path / "report/results.json").read_text())
    assert output["models"]["B"]["macro_success"] == 0.96
    assert output["pairs"]["B-C"]["mean"] == pytest.approx(0.16)
    assert output["pairs"]["B-C"]["only_left"] == 240
    assert output["pairs"]["B-C"]["only_right"] == 0
    assert output["decision"].startswith("B ")
    assert (tmp_path / "report/index.html").is_file()
    assert (tmp_path / "report/success-curves.png").stat().st_size > 1000
