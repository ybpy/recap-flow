"""Immutable rollout conversion, pool construction and per-task relabeling."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np
from openpi_client.e0.artifacts import content_id
from openpi_client.e0.artifacts import read_jsonl
from openpi_client.e0.artifacts import rewrite_columns
from openpi_client.e0.artifacts import sha256_file
from openpi_client.e0.artifacts import write_json
from openpi_client.e0.artifacts import write_jsonl
from openpi_client.e0.protocol import advantages
from openpi_client.e0.protocol import condition_labels
from openpi_client.e0.protocol import summarize_attempts
from openpi_client.e0.protocol import trajectory_labels
import pyarrow.parquet as pq

from openpi.e0.audit import scalar_column


def link_or_copy(source, target):
    # Snapshots are immutable: all label updates always create a new output directory.
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def convert_rollouts(episode_manifest, results_dir, demo_dir, output, purpose, branch, round_index, suite=None):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    expected = read_jsonl(episode_manifest)
    records = [json.loads(p.read_text()) for p in sorted((Path(results_dir) / "attempts").glob("*.json"))]
    summary = summarize_attempts(expected, records)
    if not summary["complete"]:
        raise ValueError(f"attempt set incomplete: {summary}")
    if any(row.get("purpose") not in (purpose, "smoke") for row in expected):
        raise ValueError("episode namespace cannot be repurposed as another data source")
    if purpose not in ("rollout", "development", "final"):
        raise ValueError("invalid provenance purpose")
    demo_dir, output = Path(demo_dir), Path(output)
    info = json.loads((demo_dir / "meta/info.json").read_text())
    template = json.loads((demo_dir / "e0-manifest.json").read_text())
    tasks = read_jsonl(demo_dir / "meta/tasks.jsonl")
    known = {row["task"]: row["task_index"] for row in tasks}
    defaults = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    features = {k: dict(v, shape=tuple(v["shape"])) for k, v in info["features"].items() if k not in defaults}
    dataset = LeRobotDataset.create(
        repo_id="recap-flow/rollout",
        root=output,
        fps=10,
        robot_type=info["robot_type"],
        features=features,
        use_videos=False,
        image_writer_threads=4,
        image_writer_processes=0,
    )
    for row in sorted(tasks, key=lambda x: x["task_index"]):
        dataset.meta.add_task(row["task"])
    episodes = []
    try:
        for result in sorted(records, key=lambda row: row["episode_uid"]):
            if suite and result["suite"] != suite:
                continue
            if result["policy_steps"] == 0:
                continue
            with np.load(result["trajectory"], allow_pickle=False) as archive:
                # NpzFile decompresses on every key access; materialize each array once.
                trajectory = {key: archive[key] for key in ("image", "wrist_image", "state", "actions")}
            length = len(trajectory["actions"])
            if length != result["policy_steps"]:
                raise ValueError("ledger/trajectory length mismatch")
            labels = trajectory_labels(length, result["success"])
            prompt = result["prompt"]
            if prompt not in known:
                raise ValueError(f"rollout prompt missing from audited task map: {prompt}")
            index = len(episodes)
            for t in range(length):
                frame = {key: trajectory[key][t] for key in ("image", "wrist_image", "state", "actions")}
                frame.update(intervention=np.asarray([0], np.int64), adv_ind="none", task=prompt)
                frame.update({key: np.asarray([value[t]], np.float32) for key, value in labels.items()})
                dataset.add_frame(frame)
            dataset.save_episode()
            relative = info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
            episodes.append(
                {
                    "episode_index": index,
                    "episode_uid": result["episode_uid"],
                    "task_index": known[prompt],
                    "suite": result["suite"],
                    "length": length,
                    "source": purpose,
                    "branch": branch,
                    "round": round_index,
                    "path": relative,
                    "training_seed": result["training_seed"],
                    "smoke": result["purpose"] == "smoke",
                    "success": result["success"],
                    "sha256": sha256_file(output / relative),
                }
            )
    finally:
        dataset.stop_image_writer()
    if not episodes:
        for name in ("episodes.jsonl", "episodes_stats.jsonl"):
            write_jsonl(output / "meta" / name, [])
    manifest = {
        "schema_version": 1,
        "episodes": episodes,
        "long_task_indices": template["long_task_indices"],
        "attempt_manifest_id": content_id(expected),
        "attempts": summary,
        "branch": branch,
        "round": round_index,
    }
    manifest["dataset_id"] = content_id(manifest)
    write_json(output / "e0-manifest.json", manifest)


def merge(sources, output):
    sources, output = list(map(Path, sources)), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifests = [json.loads((p / "e0-manifest.json").read_text()) for p in sources]
    training = any(e["source"] in ("demo", "rollout") for m in manifests for e in m["episodes"])
    validation = any(e["source"] in ("development", "final") for m in manifests for e in m["episodes"])
    if training and validation:
        raise ValueError("cannot mix training and evaluation trajectories")
    first = sources[0]
    info = json.loads((first / "meta/info.json").read_text())
    tasks = read_jsonl(first / "meta/tasks.jsonl")
    for source in sources[1:]:
        if read_jsonl(source / "meta/tasks.jsonl") != tasks:
            raise ValueError("task maps must agree exactly before merging")
    (output / "meta").mkdir(parents=True)
    episodes, meta_episodes, meta_stats, seen, offset = [], [], [], set(), 0
    for source, manifest in zip(sources, manifests, strict=True):
        stats = {row["episode_index"]: row for row in read_jsonl(source / "meta/episodes_stats.jsonl")}
        for episode in manifest["episodes"]:
            uid = episode["episode_uid"]
            if uid in seen:
                raise ValueError(f"duplicate trajectory in cumulative pool: {uid}")
            seen.add(uid)
            index, length = len(episodes), episode["length"]
            relative = info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source == first:
                link_or_copy(source / episode["path"], target)
                digest = episode["sha256"]
            else:
                table = pq.read_table(source / episode["path"])
                table = rewrite_columns(
                    table, {"episode_index": [index] * length, "index": np.arange(offset, offset + length)}
                )
                pq.write_table(table, target)
                digest = sha256_file(target)
            episodes.append(dict(episode, episode_index=index, path=relative, sha256=digest))
            meta_episodes.append(
                {"episode_index": index, "length": length, "tasks": [tasks[episode["task_index"]]["task"]]}
            )
            statistic = dict(stats[episode["episode_index"]], episode_index=index)
            # Index statistics are metadata only; recompute after renumbering.
            for key, values in (
                ("episode_index", np.full(length, index)),
                ("index", np.arange(offset, offset + length)),
            ):
                if key in statistic["stats"]:
                    statistic["stats"][key] = {
                        "min": [int(values.min())],
                        "max": [int(values.max())],
                        "mean": [float(values.mean())],
                        "std": [float(values.std())],
                        "count": [length],
                    }
            meta_stats.append(statistic)
            offset += length
    info.update(
        total_episodes=len(episodes),
        total_frames=offset,
        total_chunks=(len(episodes) + info["chunks_size"] - 1) // info["chunks_size"],
        splits={"train": f"0:{len(episodes)}"},
    )
    write_json(output / "meta/info.json", info)
    write_jsonl(output / "meta/tasks.jsonl", tasks)
    write_jsonl(output / "meta/episodes.jsonl", meta_episodes)
    write_jsonl(output / "meta/episodes_stats.jsonl", meta_stats)
    result = {
        "schema_version": 1,
        "sources": [m["dataset_id"] for m in manifests],
        "source_manifests": [
            {
                "path": str(p / "e0-manifest.json"),
                "dataset_id": m["dataset_id"],
                "sha256": sha256_file(p / "e0-manifest.json"),
            }
            for p, m in zip(sources, manifests, strict=True)
        ],
        "episodes": episodes,
        "long_task_indices": manifests[0]["long_task_indices"],
    }
    result["dataset_id"] = content_id(result)
    write_json(output / "e0-manifest.json", result)


def relabel(source, predictions, output):
    source, predictions, output = map(Path, (source, predictions, output))
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / "e0-manifest.json").read_text())
    pred = json.loads((predictions / "predictions.json").read_text())
    if pred["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("predictions belong to a different data snapshot")
    by_uid = {e["episode_uid"]: e for e in pred["episodes"]}
    advantages_all, tasks_all, expert_all, slices = [], [], [], []
    offset = 0
    for episode in manifest["episodes"]:
        table = pq.read_table(source / episode["path"])
        length = len(table)
        expert = np.ones(length, bool) if episode["source"] == "demo" else scalar_column(table, "intervention") != 0
        if episode["source"] not in ("demo", "rollout"):
            raise ValueError("validation/final trajectories cannot enter training percentile pool")
        if episode["source"] == "demo":
            adv = np.zeros(length)
        else:
            # Require new predictions for every accumulated autonomous trajectory.
            prediction = by_uid[episode["episode_uid"]]
            values = np.load(predictions / prediction["path"], allow_pickle=False)
            adv = advantages(scalar_column(table, "reward_label"), values)
        advantages_all.extend(adv)
        tasks_all.extend([episode["task_index"]] * length)
        expert_all.extend(expert)
        slices.append(slice(offset, offset + length))
        offset += length
    labels, diagnostics = condition_labels(tasks_all, advantages_all, expert_all)
    shutil.copytree(source / "meta", output / "meta")
    records = []
    for episode, selection in zip(manifest["episodes"], slices, strict=True):
        target = output / episode["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if episode["source"] == "demo":
            link_or_copy(source / episode["path"], target)
            digest = episode["sha256"]
        else:
            table = pq.read_table(source / episode["path"])
            table = rewrite_columns(table, {"adv_ind": labels[selection].tolist()})
            pq.write_table(table, target)
            digest = sha256_file(target)
        records.append(dict(episode, sha256=digest))
    result = dict(
        manifest,
        episodes=records,
        parent_dataset_id=manifest["dataset_id"],
        critic_checkpoint=pred["checkpoint"],
        prediction_manifest_id=content_id(pred),
        label_protocol="per-task percentile70 >=; demo/intervention excluded; all autonomous relabeled",
    )
    result.pop("dataset_id", None)
    result["dataset_id"] = content_id(result)
    write_json(output / "e0-manifest.json", result)
    write_json(output / "label-diagnostics.json", diagnostics)
    np.savez_compressed(
        output / "advantage-diagnostics.npz",
        advantage=np.asarray(advantages_all),
        task_index=np.asarray(tasks_all),
        expert=np.asarray(expert_all),
    )
