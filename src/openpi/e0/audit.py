"""Read-only raw/converted audit; writes derived demo snapshot and frozen stats."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
from openpi_client.e0.artifacts import content_id
from openpi_client.e0.artifacts import read_jsonl
from openpi_client.e0.artifacts import rewrite_columns
from openpi_client.e0.artifacts import sha256_file
from openpi_client.e0.artifacts import write_json
from openpi_client.e0.protocol import trajectory_labels
from PIL import Image
import pyarrow.parquet as pq


def signature(state, actions, prompt):
    h = hashlib.sha256()
    for value in (state, actions):
        x = np.ascontiguousarray(value, dtype="<f4")
        h.update(str(x.shape).encode())
        h.update(x.tobytes())
    h.update(prompt.encode())
    return h.hexdigest()


def image_array(value):
    if isinstance(value, dict):
        value = value.get("bytes") or Path(value["path"]).read_bytes()
    with Image.open(io.BytesIO(value)) as im:
        im.load()
        if im.mode != "RGB" or im.size != (256, 256):
            raise ValueError(f"unexpected image format: {im.mode}, {im.size}")
        return np.asarray(im)


def scalar_column(table, key):
    return np.asarray(table[key].to_pylist()).reshape(len(table), -1)[:, 0]


def run(source, rlds_root, output, report_dir):
    source, output, report_dir = map(Path, (source, output, report_dir))
    report_dir.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"snapshot already exists: {output}")
    info = json.loads((source / "meta/info.json").read_text())
    tasks = {row["task_index"]: row["task"] for row in read_jsonl(source / "meta/tasks.jsonl")}
    episode_meta = read_jsonl(source / "meta/episodes.jsonl")
    raw = {}
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)
    raw_inventory = []
    for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial"):
        directory = Path(rlds_root) / (suite + "_no_noops") / "1.0.0"
        builder = tfds.builder_from_directory(str(directory))
        # Use TFDS's own decoder, exactly as the openpi converter does.
        # Pillow and TensorFlow can round JPEG pixels differently.
        ds = builder.as_dataset(split="train", shuffle_files=False)
        count = 0
        for episode in ds:
            steps = next(iter(episode["steps"].batch(10000)))
            if float(steps["reward"][-1]) != 1.0:
                raise ValueError(
                    "standard demo source contains a non-success terminal reward; inspect without filtering"
                )
            state = steps["observation"]["state"].numpy()
            actions = steps["action"].numpy()
            texts = steps["language_instruction"].numpy()
            prompt = texts[0].decode()
            if any(x.decode() != prompt for x in texts):
                raise ValueError("language changes within raw demo")
            key = signature(state, actions, prompt)
            picks = sorted({0, len(state) // 2, len(state) - 1})
            pixels = {}
            for camera in ("image", "wrist_image"):
                decoded_raw = steps["observation"][camera].numpy()
                if decoded_raw.shape != (len(state), 256, 256, 3) or decoded_raw.dtype != np.uint8:
                    raise ValueError("unexpected raw decoded image schema")
                pixels[camera] = {i: hashlib.sha256(decoded_raw[i].tobytes()).hexdigest() for i in picks}
            raw.setdefault(key, []).append({"suite": suite, "pixels": pixels, "length": len(state)})
            count += 1
        raw_inventory.append(
            {
                "suite": suite,
                "episodes": count,
                "files": [
                    {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                    for p in sorted(directory.glob("*tfrecord*"))
                ],
            }
        )
        write_json(report_dir / "audit-progress.json", {"raw": raw_inventory})
        print(f"raw audit {suite}: {count} episodes", flush=True)
    output.mkdir(parents=True)
    shutil.copytree(source / "meta", output / "meta")
    episodes, norms = [], {"state": [], "actions": []}
    changed_labels, decoded = 0, 0
    for episode in episode_meta:
        index, length = episode["episode_index"], episode["length"]
        relative = info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
        path = source / relative
        table = pq.read_table(path)
        if len(table) != length:
            raise ValueError(f"length mismatch: {path}")
        if not np.array_equal(scalar_column(table, "frame_index"), np.arange(length)):
            raise ValueError(f"non-contiguous frame indices: {path}")
        if not np.all(scalar_column(table, "episode_index") == index):
            raise ValueError(f"episode index mismatch: {path}")
        task_indices = scalar_column(table, "task_index")
        if len(np.unique(task_indices)) != 1:
            raise ValueError("multiple tasks in demo")
        task_index = int(task_indices[0])
        prompt = tasks[task_index]
        state = np.asarray(table["state"].to_pylist(), np.float32)
        actions = np.asarray(table["actions"].to_pylist(), np.float32)
        if state.shape != (length, 8) or actions.shape != (length, 7):
            raise ValueError(f"bad state/action shape: {path}")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError(f"nonfinite state/action: {path}")
        key = signature(state, actions, prompt)
        matches = raw.get(key, [])
        if not matches:
            raise ValueError(f"converted demo not matched exactly to raw state/actions/text: {path}")
        match = matches.pop()
        for camera in ("image", "wrist_image"):
            for frame, value in enumerate(table[camera].to_pylist()):
                pixels = image_array(value)
                decoded += 1
                expected = match["pixels"][camera].get(frame)
                if expected and hashlib.sha256(pixels.tobytes()).hexdigest() != expected:
                    raise ValueError(f"pixel mismatch: {path}, {camera}, frame {frame}")
        labels = trajectory_labels(length, success=True)
        for name, expected in labels.items():
            if name in table.column_names and not np.allclose(scalar_column(table, name), expected, atol=2e-7):
                changed_labels += 1
        # All source trajectories matched standard no_noops demos; fixed positive.
        labels["adv_ind"] = ["positive"] * length
        table = rewrite_columns(table, labels)
        # Reconstruct missing HF image feature metadata without changing image payload.
        import datasets

        features = datasets.Features.from_arrow_schema(table.schema)
        for camera in ("image", "wrist_image"):
            features[camera] = datasets.Image()
        metadata = dict(table.schema.metadata or {})
        metadata[b"huggingface"] = features.arrow_schema.metadata[b"huggingface"]
        table = table.replace_schema_metadata(metadata)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)
        episodes.append(
            {
                "episode_index": index,
                "episode_uid": f"demo/{match['suite']}/{key}",
                "task_index": task_index,
                "suite": match["suite"],
                "length": length,
                "source": "demo",
                "path": relative,
                "content_signature": key,
                "source_sha256": sha256_file(path),
                "sha256": sha256_file(target),
            }
        )
        norms["state"].append(state)
        # Match openpi action chunks, including its repeated final action padding.
        ids = np.minimum(np.arange(length)[:, None] + np.arange(10)[None, :], length - 1)
        norms["actions"].append(actions[ids].reshape(-1, 7))
        if len(episodes) % 50 == 0:
            write_json(
                report_dir / "audit-progress.json", {"converted_episodes": len(episodes), "images_decoded": decoded}
            )
            print(f"converted audit {len(episodes)}/{len(episode_meta)}", flush=True)
    unmatched = sum(len(value) for value in raw.values())
    if unmatched:
        raise ValueError(f"{unmatched} raw episodes absent from converted source")
    long_tasks = sorted({e["task_index"] for e in episodes if e["suite"] == "libero_10"})
    manifest = {
        "schema_version": 1,
        "source": str(source),
        "raw_root": str(rlds_root),
        "episodes": episodes,
        "long_task_indices": long_tasks,
        "raw_inventory": raw_inventory,
    }
    manifest["dataset_id"] = content_id(manifest)
    write_json(output / "e0-manifest.json", manifest)
    from openpi.shared import normalize

    result = {}
    for name, arrays in norms.items():
        stats = normalize.RunningStats()
        for array in arrays:
            stats.update(array)
        result[name] = stats.get_statistics()
    norm_dir = report_dir.parent / "assets" / "libero"
    if norm_dir.exists():
        raise FileExistsError(f"frozen norm directory exists: {norm_dir}")
    normalize.save(norm_dir, result)
    write_json(
        norm_dir / "provenance.json",
        {
            "dataset_id": manifest["dataset_id"],
            "horizon": 10,
            "action_weighting": "all frame-anchored H=10 chunks; endpoint repeats, as openpi loader",
            "frames": sum(e["length"] for e in episodes),
            "sha256": sha256_file(norm_dir / "norm_stats.json"),
        },
    )
    write_json(
        report_dir / "data-audit.json",
        {
            "complete": True,
            "dataset_id": manifest["dataset_id"],
            "episodes": len(episodes),
            "frames": sum(e["length"] for e in episodes),
            "decoded_images": decoded,
            "stratified_pixel_comparison": "first/middle/last, both cameras, every episode",
            "corrected_label_columns": changed_labels,
            "snapshot": str(output),
            "norm_dir": str(norm_dir),
        },
    )
    return manifest
