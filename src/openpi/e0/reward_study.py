"""R1 reward definitions and immutable, frame-aligned critic supervision."""

import json
from pathlib import Path

import numpy as np
from openpi_client.e0.artifacts import content_id, sha256_file, write_json
from openpi_client.e0.protocol import LIMITS, trajectory_labels

METHODS = ("legacy_episode", "task_pool_max", "global_pool_max")
MODES = ("all_backbones", "none")


def fit_scales(pool):
    maxima = {}
    for ep in pool["episodes"]:
        if ep["source"] not in {"demo", "rollout"} or ep["length"] < 1:
            raise ValueError("fit reward scales only on positive-length training trajectories")
        task = str(ep["task_index"])
        maxima[task] = max(maxima.get(task, 0), ep["length"])
    scale = max(1.5, *(0.5 + (LIMITS["libero_10"] - 1) / maxima[str(j)] for j in pool["long_task_indices"]))
    return dict(
        train_dataset_id=pool["dataset_id"], task_max=maxima, global_max=max(maxima.values()), head_scale=scale
    )


def targets(method, length, success, task, scales):
    if method == "legacy_episode":
        labels = trajectory_labels(length, success)
        return np.column_stack((labels["reward_label"], labels["value_label"]))
    if method not in METHODS or length < 1:
        raise ValueError("invalid protocol or episode length")
    denom = scales["task_max"][str(task)] if method == "task_pool_max" else scales["global_max"]
    rewards = np.full(length, -1.0 / denom, np.float64)
    rewards[-1] = 0 if success else -0.5
    returns = np.cumsum(rewards[::-1])[::-1]
    return np.column_stack((rewards, returns)).astype(np.float32)


def load_dataset(directory):
    data = json.loads((Path(directory) / "e0-manifest.json").read_text())
    if data["dataset_id"] != content_id({k: v for k, v in data.items() if k != "dataset_id"}):
        raise ValueError("damaged dataset manifest")
    return data


def save_labels(directory, method, scales, datasets):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(method=method, scales=scales, scale=1.0 if method == METHODS[0] else scales["head_scale"], splits={})
    for split, data in datasets.items():
        rows, index, offset = [], [], 0
        for ep in data["episodes"]:
            success = True if ep["source"] == "demo" else ep["success"]
            a = targets(method, ep["length"], success, ep["task_index"], scales)
            rows.append(a)
            index.append([ep["episode_uid"], offset, len(a)])
            offset += len(a)
        values = np.concatenate(rows)
        if not np.isfinite(values).all() or values[:, 1].min() < -record["scale"] - 1e-6 or values[:, 1].max() > 0:
            raise ValueError(f"critic support exceeded: {method}/{split}")
        file = directory / (split + ".npy")
        np.save(file, values, allow_pickle=False)
        record["splits"][split] = dict(dataset_id=data["dataset_id"], episodes=index, sha256=sha256_file(file))
    record["protocol_id"] = content_id(record)
    write_json(directory / "protocol.json", record)
    return record


def load_labels(spec, data, role):
    path = Path(spec["reward_protocol"])
    record = json.loads(path.read_text())
    identity = record.pop("protocol_id")
    if identity != content_id(record) or identity != spec["reward_protocol_id"]:
        raise ValueError("reward protocol checksum mismatch")
    if record["scales"]["train_dataset_id"] != spec["dataset_id"]:
        raise ValueError("reward scale fit to a different training pool")
    entry = record["splits"][role]
    file = path.parent / (role + ".npy")
    if (
        entry["dataset_id"] != data["dataset_id"]
        or sha256_file(file) != entry["sha256"]
        or [(u, n) for u, _, n in entry["episodes"]] != [(e["episode_uid"], e["length"]) for e in data["episodes"]]
    ):
        raise ValueError("label file does not match dataset/role")
    return np.load(file, mmap_mode="r", allow_pickle=False), record["scale"]


class ValueDataset:
    def __init__(self, dataset, physical_targets, scale):
        self.dataset = dataset
        self.encoded = np.asarray(physical_targets[:, 1] / scale, dtype=np.float32)
        if len(self.dataset) != len(self.encoded) or np.any((self.encoded < -1) | (self.encoded > 0)):
            raise ValueError("encoded critic target length or bounds invalid")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        return dict(row, value=np.full_like(row["value"], self.encoded[index], dtype=np.float32))
