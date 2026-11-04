"""Materialize critic study supervision and actor conditions in immutable snapshots."""

import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq
from openpi_client.e0.artifacts import content_id, read_jsonl, rewrite_columns, sha256_file, write_json, write_jsonl
from openpi_client.e0.protocol import condition_labels

from openpi.e0.audit import scalar_column
from openpi.e0.critic_report import parts, read_thresholds
from openpi.e0.reward_study import CLIP_METHODS, load_dataset, load_labels


def relabel_study(source, predictions, output, critic_manifest, thresholds):
    source, predictions, output = map(Path, (source, predictions, output))
    if output.exists():
        raise FileExistsError(output)
    data = load_dataset(source)
    spec = json.loads(Path(critic_manifest).read_text())
    if spec["manifest_id"] != content_id({k: v for k, v in spec.items() if k != "manifest_id"}):
        raise ValueError("critic manifest checksum mismatch")
    targets, scale = load_labels(spec, data, "train")
    protocol = json.loads(Path(spec["reward_protocol"]).read_text())
    if protocol["method"] not in CLIP_METHODS or scale != 1 or targets.shape[1] != 3:
        raise ValueError("actor comparison requires direct-clip supervision")
    pred = json.loads((predictions / "predictions.json").read_text())
    if (
        pred["dataset_id"] != data["dataset_id"]
        or pred["manifest_id"] != spec["manifest_id"]
        or pred["reward_protocol_id"] != spec["reward_protocol_id"]
        or pred["role"] != "train"
        or pred["prediction_units"] != "reward"
    ):
        raise ValueError("prediction identity or training role differs")
    checkpoint = json.loads((Path(pred["checkpoint"]) / "e0-checkpoint.json").read_text())
    if checkpoint["manifest_id"] != spec["manifest_id"] or checkpoint["parameter_kind"] != "EMA":
        raise ValueError("prediction checkpoint differs from critic")
    frozen = read_thresholds(thresholds, spec, pred["checkpoint"], data["dataset_id"])
    expected = {e["episode_uid"] for e in data["episodes"] if e["source"] == "rollout"}
    records = {e["episode_uid"]: e for e in pred["episodes"]}
    if set(records) != expected or len(records) != len(pred["episodes"]):
        raise ValueError("predictions must cover every autonomous episode exactly once")
    n = sum(e["length"] for e in data["episodes"])
    expert = np.ones(n, bool)
    task = np.empty(n, np.int64)
    value = np.full(n, np.nan, np.float32)
    window, delta, advantage = [np.full(n, np.nan, np.float64) for _ in range(3)]
    offset, hashes = 0, []
    for ep in data["episodes"]:
        if ep["source"] not in ("demo", "rollout"):
            raise ValueError("only training trajectories may be relabeled")
        path = source / ep["path"]
        if sha256_file(path) != ep["sha256"]:
            raise ValueError(f"source episode changed: {ep['episode_uid']}")
        sl = slice(offset, offset + ep["length"])
        task[sl] = ep["task_index"]
        if ep["source"] == "rollout":
            row = records[ep["episode_uid"]]
            vpath = predictions / row["path"]
            v = np.load(vpath, allow_pickle=False)
            if row["length"] != ep["length"] or v.shape != (ep["length"],) or not np.isfinite(v).all():
                raise ValueError("prediction frame alignment or finiteness differs")
            if v.min() < -1.000001 or v.max() > 0.000001:
                raise ValueError("direct-clip prediction support exceeded")
            value[sl] = v
            window[sl], delta[sl], advantage[sl] = parts(targets[sl, 0], v)
            expert[sl] = scalar_column(pq.read_table(path, columns=["intervention"]), "intervention") != 0
            hashes.append(dict(episode_uid=ep["episode_uid"], sha256=sha256_file(vpath)))
        offset = sl.stop
    labels, diagnostics = condition_labels(task, advantage, expert)
    computed = {k: v["threshold"] for k, v in diagnostics.items()}
    if computed != frozen:
        raise ValueError("recomputed training thresholds differ from frozen critic review")
    for key, threshold in frozen.items():
        mask = (task == int(key)) & ~expert
        np.testing.assert_array_equal(labels[mask] == "positive", advantage[mask] >= threshold)
    shutil.copytree(source / "meta", output / "meta")
    stats = {r["episode_index"]: r for r in read_jsonl(source / "meta/episodes_stats.jsonl")}
    episodes, offset = [], 0
    changed_columns = ["reward_label", "value_label", "adv_ind"]
    for ep in data["episodes"]:
        sl = slice(offset, offset + ep["length"])
        table = pq.read_table(source / ep["path"])
        replacements = dict(reward_label=targets[sl, 0], value_label=targets[sl, 1], adv_ind=labels[sl])
        updated = rewrite_columns(table, replacements)
        target = output / ep["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(updated, target)
        saved = pq.read_table(target)
        if not table.drop(changed_columns).equals(saved.drop(changed_columns), check_metadata=True):
            raise ValueError("relabel changed observations/actions or their Arrow metadata")
        for column in ("reward_label", "value_label"):
            arr = scalar_column(saved, column).astype(np.float64)
            np.testing.assert_array_equal(arr, replacements[column])
            stats[ep["episode_index"]]["stats"][column] = dict(
                count=[len(arr)],
                min=[float(arr.min())],
                max=[float(arr.max())],
                mean=[float(arr.mean())],
                std=[float(arr.std())],
            )
        if saved["adv_ind"].to_pylist() != labels[sl].tolist():
            raise ValueError("persisted actor conditions differ")
        episodes.append(dict(ep, sha256=sha256_file(target)))
        offset = sl.stop
    write_jsonl(output / "meta/episodes_stats.jsonl", [stats[e["episode_index"]] for e in episodes])
    threshold_record = json.loads(Path(thresholds).read_text())
    binding = dict(
        parent_dataset_id=data["dataset_id"],
        critic_checkpoint=pred["checkpoint"],
        critic_manifest_id=spec["manifest_id"],
        reward_protocol_id=spec["reward_protocol_id"],
        reward_protocol=spec["reward_protocol"],
        prediction_manifest_id=content_id(pred),
        prediction_files_id=content_id(hashes),
        threshold_id=threshold_record["threshold_id"],
        label_protocol="direct-clip; task percentile70 >=; experts positive; lookahead50",
    )
    step_threshold = np.array([frozen.get(str(t), np.nan) for t in task])
    np.savez_compressed(
        output / "advantage-diagnostics.npz",
        reward=targets[:, 0],
        return_before_clip=targets[:, 2],
        value_target=targets[:, 1],
        value_pred=value,
        window_reward=window,
        value_delta=delta,
        advantage=advantage,
        task_index=task,
        expert=expert,
        positive=labels == "positive",
        threshold=step_threshold,
    )
    binding["diagnostics_sha256"] = sha256_file(output / "advantage-diagnostics.npz")
    write_json(output / "thresholds.json", threshold_record)
    write_json(output / "label-diagnostics.json", diagnostics)
    write_json(output / "prediction-files.json", hashes)
    result = dict(data, episodes=episodes, **binding)
    result.pop("dataset_id")
    result["dataset_id"] = content_id(result)
    write_json(output / "e0-manifest.json", result)


def validate_label_binding(directory, manifest):
    """Keep actor training and restoration tied to the audited conditions."""
    if not manifest.get("reward_protocol_id"):
        return
    path = Path(directory)
    thresholds = json.loads((path / "thresholds.json").read_text())
    identity = thresholds.pop("threshold_id")
    if (
        identity != content_id(thresholds)
        or identity != manifest["threshold_id"]
        or thresholds["manifest_id"] != manifest["critic_manifest_id"]
        or thresholds["reward_protocol_id"] != manifest["reward_protocol_id"]
        or thresholds["checkpoint"] != manifest["critic_checkpoint"]
        or thresholds["train_dataset_id"] != manifest["parent_dataset_id"]
        or sha256_file(path / "advantage-diagnostics.npz") != manifest["diagnostics_sha256"]
    ):
        raise ValueError("actor dataset/critic/threshold binding differs")
