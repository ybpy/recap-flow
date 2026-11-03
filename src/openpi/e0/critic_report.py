"""Paired trajectory-level critic review, using thresholds fitted only on training rollouts."""

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from openpi_client.e0.artifacts import content_id, write_json
from openpi_client.e0.protocol import LOOKAHEAD

from openpi.e0.reward_study import CLIP_METHODS, METHODS, MODES, load_dataset, load_labels


def read(path):
    return json.loads(Path(path).read_text())


def parts(rewards, values):
    v = np.asarray(values, dtype=np.float64)
    ends = np.minimum(np.arange(len(v)) + LOOKAHEAD, len(v))
    sums = np.r_[0, np.cumsum(rewards, dtype=np.float64)]
    window = sums[ends] - sums[: len(v)]
    delta = np.r_[v, 0.0][ends] - v
    return window, delta, window + delta


def training_clocks(data, labels):
    sums, counts = defaultdict(float), defaultdict(int)
    offset = 0
    for ep in data["episodes"]:
        task, n = ep["task_index"], ep["length"]
        target = labels[offset : offset + n, 1]
        for start in range(0, n, 50):
            chunk = target[start : start + 50]
            for key in ((task, start // 50), (task, -1)):
                sums[key] += chunk.sum(dtype=np.float64)
                counts[key] += len(chunk)
        offset += n
    return {k: sums[k] / counts[k] for k in sums}


def read_thresholds(path, spec, checkpoint, train_id):
    record = read(path)
    checksum = record.pop("threshold_id")
    if (
        checksum != content_id(record)
        or record["manifest_id"] != spec["manifest_id"]
        or record["reward_protocol_id"] != spec["reward_protocol_id"]
        or record["checkpoint"] != checkpoint
        or record["train_dataset_id"] != train_id
    ):
        raise ValueError("threshold binding differs from model/protocol/training pool")
    return record["thresholds"]


def analyze_model(base, dest, data):
    spec = read(base / "training.json")
    name = base.name
    model_dir = dest / name
    model_dir.mkdir()
    selection = read(base / "selection.json")
    thresholds = {}
    frames = {}
    statistics = {}
    for role in ("train", "review"):
        manifest = data[role]
        labels, head_scale = load_labels(spec, manifest, role)
        prediction = read(base / role / "predictions.json")
        if (
            prediction["reward_protocol_id"] != spec["reward_protocol_id"]
            or prediction["role"] != role
            or prediction["checkpoint"] != selection["checkpoint"]
            or prediction["dataset_id"] != manifest["dataset_id"]
            or prediction["manifest_id"] != spec["manifest_id"]
            or prediction["prediction_units"] != "reward"
        ):
            raise ValueError(f"wrong predictions for {name}/{role}")
        by_uid = {e["episode_uid"]: e for e in prediction["episodes"]}
        offset, per_task = 0, defaultdict(list)
        if role == "train":
            clock = training_clocks(manifest, labels)
        else:
            thresholds = read_thresholds(
                model_dir / "thresholds.json", spec, selection["checkpoint"], data["train"]["dataset_id"]
            )
        for ep in manifest["episodes"]:
            n, uid, task = ep["length"], ep["episode_uid"], ep["task_index"]
            episode_labels = labels[offset : offset + n]
            rewards, target = episode_labels[:, :2].T
            offset += n
            if ep["source"] == "demo":
                continue
            record = by_uid.pop(uid)
            value = np.load(base / role / record["path"], allow_pickle=False)
            ce = np.load(base / role / record["ce_path"], allow_pickle=False)
            if record["length"] != n or value.shape != (n,) or ce.shape != (n,) or not np.isfinite([value, ce]).all():
                raise ValueError(f"unaligned or non-finite {uid}")
            if episode_labels.shape[1] == 3 and (head_scale != 1 or value.min() < -1.000001 or value.max() > 0.000001):
                raise ValueError("direct-clip predictor has invalid scale or output")
            window, delta, advantage = parts(rewards, value)
            per_task[task].append(advantage)
            if role == "review":
                if episode_labels.shape[1] == 3:
                    before = episode_labels[:, 2]
                elif name.startswith("legacy_episode-"):
                    before = np.cumsum(rewards[::-1], dtype=np.float64)[::-1].astype(np.float32)
                else:
                    before = target
                threshold = thresholds[str(task)]
                positive = advantage >= threshold
                time_value = np.asarray([clock.get((task, t // 50), clock[task, -1]) for t in range(n)])
                error, naive = value - target, time_value - target
                file = f"episode-{ep['episode_index']:04d}.npz"
                np.savez_compressed(
                    model_dir / file,
                    reward=rewards,
                    target=target,
                    value=value,
                    return_before_clip=before,
                    value_target=target,
                    head_target=target / head_scale,
                    value_pred=value,
                    ce=ce,
                    clock=time_value,
                    window_reward=window,
                    value_delta=delta,
                    advantage=advantage,
                    positive=positive,
                    threshold=np.float64(threshold),
                )
                frames[uid] = dict(
                    path=file,
                    advantage=advantage,
                    positive=positive,
                    error=error,
                    ce=ce,
                    value=value,
                    before=before,
                    threshold=threshold,
                )
                stats = dict(
                    ce=float(ce.mean()),
                    mae=float(np.abs(error).mean()),
                    mse=float(np.mean(error**2)),
                    clock_mse=float(np.mean(naive**2)),
                    positive=float(positive.mean()),
                    terminal50_mae=float(np.abs(error[-50:]).mean()),
                    terminal50_clock_mae=float(np.abs(naive[-50:]).mean()),
                    clock_mae=float(np.abs(naive).mean()),
                )
                for j, chunk in enumerate(np.array_split(positive, 4)):
                    if len(chunk):
                        stats[f"positive_quarter_{j+1}"] = float(chunk.mean())
                statistics[uid] = stats
        if by_uid:
            raise ValueError("extra predictions")
        if role == "train":
            thresholds = {
                str(k): float(np.percentile(np.concatenate(v), 70, method="linear")) for k, v in per_task.items()
            }
            training_positive = {
                str(k): float(np.mean(np.concatenate(v) >= thresholds[str(k)])) for k, v in per_task.items()
            }
            binding = dict(
                manifest_id=spec["manifest_id"],
                reward_protocol_id=spec["reward_protocol_id"],
                checkpoint=selection["checkpoint"],
                train_dataset_id=manifest["dataset_id"],
                selection_dataset_id=selection["validation_dataset_id"],
                thresholds=thresholds,
                train_positive_fraction=training_positive,
            )
            binding["threshold_id"] = content_id(binding)
            write_json(model_dir / "thresholds.json", binding)
    write_json(
        model_dir / "frames.json",
        dict(
            threshold_id=binding["threshold_id"],
            review_dataset_id=data["review"]["dataset_id"],
            episodes={u: r["path"] for u, r in frames.items()},
        ),
    )
    return frames, statistics


def frame_rates(frames, attempts, selected):
    counts = defaultdict(lambda: [0, 0])
    for uid in selected:
        task = attempts[uid]["task_id"]
        label = frames[uid]["positive"]
        counts[task][0] += int(label.sum())
        counts[task][1] += len(label)
    return {str(t): dict(positive=p, frames=n, fraction=p / n) for t, (p, n) in sorted(counts.items())}


def paired_metrics(left, right):
    a, b = left["positive"], right["positive"]
    union = np.sum(a | b)
    out = dict(
        flip=float(np.mean(a != b)),
        jaccard=float(np.sum(a & b) / union) if union else None,
        ce_difference=float(np.mean(left["ce"] - right["ce"])),
        mae_difference=float(np.mean(np.abs(left["error"]) - np.abs(right["error"]))),
    )
    x, y = left["advantage"], right["advantage"]
    out["spearman"] = float(spearmanr(x, y).statistic) if len(x) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0 else None
    for j, chunk in enumerate(np.array_split(a != b, 4)):
        if len(chunk):
            out[f"flip_quarter_{j+1}"] = float(chunk.mean())
    return out


def estimates(metrics, attempts, seed=42):
    """Equal weight to trajectories within task, and to tasks in the macro average."""
    import warnings

    rng = np.random.default_rng(seed)
    result, macro = {}, defaultdict(list)
    for task in sorted({r["task_id"] for r in attempts.values()}):
        ids = sorted(u for u, r in attempts.items() if r["task_id"] == task and u in metrics)
        if not ids:
            continue
        draws = rng.integers(0, len(ids), (1000, len(ids)))
        task_result = {}
        for key in sorted({k for uid in ids for k in metrics[uid]}):
            a = np.array([metrics[uid].get(key) for uid in ids], dtype=float)
            if not np.isfinite(a).any():
                task_result[key] = None
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                boot = np.nanmean(a[draws], axis=1)
            mean = float(np.nanmean(a))
            task_result[key] = dict(
                mean=mean, ci95=np.nanpercentile(boot, [2.5, 97.5]).tolist(), episodes=int(np.isfinite(a).sum())
            )
            macro[key].append(np.r_[mean, boot])
        result[str(task)] = task_result
    overall = {}
    for key, rows in macro.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            avg = np.nanmean(rows, axis=0)
        overall[key] = dict(mean=float(avg[0]), ci95=np.nanpercentile(avg[1:], [2.5, 97.5]).tolist(), tasks=len(rows))
    return dict(by_task=result, macro=overall)


def report(work, dest, parent=None):
    from openpi.e0.critic_gallery import gallery

    dest.mkdir(parents=True, exist_ok=True)
    source = parent or work
    data = {k: load_dataset(source / d) for k, d in (("train", "pool"), ("review", "review"))}
    attempts = {
        r["episode_uid"]: r for r in (read(p) for p in (source / "review-evaluation/results/attempts").glob("*.json"))
    }
    ids = {e["episode_uid"] for e in data["review"]["episodes"]}
    if ids != {u for u, r in attempts.items() if r["policy_steps"]}:
        raise ValueError("review ledger/frame coverage differs")
    names = [f"{m}-{f}" for m in (METHODS + CLIP_METHODS if parent else METHODS) for f in MODES]
    summaries, all_frames = {}, {}
    for name in names:
        base = (source if name.split("-")[0] in METHODS else work) / name
        frames, metrics = analyze_model(base, dest, data)
        if frames.keys() != ids:
            raise ValueError("incomplete paired inference")
        all_frames[name] = frames
        strata = {
            "all": ids,
            "success": [u for u in ids if attempts[u]["success"]],
            "failure": [u for u in ids if not attempts[u]["success"]],
            "timeout": [u for u in ids if not attempts[u]["success"] and attempts[u]["policy_steps"] == 520],
        }
        summaries[name] = {g: estimates({u: metrics[u] for u in selected}, attempts) for g, selected in strata.items()}
        for g, selected in strata.items():
            summaries[name][g]["observed_frame_fraction"] = frame_rates(frames, attempts, selected)
        summaries[name]["selection"] = read(base / "selection.json")
    pairs = {}
    for l, r in combinations(names, 2):
        same_method = l.split("-")[0] == r.split("-")[0]
        if not (same_method or l.split("-")[1] == r.split("-")[1]):
            continue
        paired = {u: paired_metrics(all_frames[l][u], all_frames[r][u]) for u in ids}
        if not same_method:
            for item in paired.values():
                item.pop("ce_difference")
                item.pop("mae_difference")
        comparison = estimates(paired, attempts)
        comparison["by_outcome"] = {
            g: estimates({u: paired[u] for u in strata[g]}, attempts) for g in ("success", "failure", "timeout")
        }
        comparison["pooled_frame_metrics"] = {}
        for task in sorted({attempts[u]["task_id"] for u in ids}):
            us = sorted(u for u in ids if attempts[u]["task_id"] == task)
            a, b = [np.concatenate([all_frames[n][u]["positive"] for u in us]) for n in (l, r)]
            x, y = [np.concatenate([all_frames[n][u]["advantage"] for u in us]) for n in (l, r)]
            comparison["pooled_frame_metrics"][str(task)] = dict(
                frames=len(a),
                flip=float(np.mean(a != b)),
                jaccard=float((a & b).sum() / (a | b).sum()) if (a | b).any() else None,
                spearman=float(spearmanr(x, y).statistic) if np.ptp(x) and np.ptp(y) else None,
            )
        pairs[l + " vs " + r] = comparison
    summary = dict(
        models=summaries,
        pairs=pairs,
        attempts=len(attempts),
        episodes_with_frames=len(ids),
        zero_action_attempts=sorted(set(attempts) - ids),
        bootstrap="1000 paired episode resamples within task; task macro",
        target_differences=read(work / "target-differences.json"),
    )
    write_json(dest / "study.json", summary)
    prior_gallery = None
    if parent:
        from openpi.e0.clip_comparison import compare

        compare(dest, all_frames, attempts, summary)
        prior_gallery = read(parent / "report/gallery.json")
    gallery(dest, names, attempts, all_frames, summary, prior_gallery=prior_gallery)
