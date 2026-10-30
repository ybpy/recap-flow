"""Predeclared diagnostic samples and analysis, isolated from E0 training."""

import itertools
import json
from pathlib import Path

import numpy as np
from openpi_client.e0.artifacts import content_id, write_json
from openpi_client.e0.protocol import condition_labels, stable_seed


def prepare(results, namespace, output):
    if not namespace.startswith("research/"):
        raise ValueError("use research/ namespace")
    paths = sorted((Path(results) / "attempts").glob("*.json"))
    records = [(p, json.loads(p.read_text())) for p in paths]
    selected = {(r["task_id"], r["initial_state_id"]): (p, r) for p, r in records
                if r["suite"] == "libero_10" and r["purpose"] == "development"
                and r["group"] == 0 and r["initial_state_id"] < 10}
    if set(selected) != set(itertools.product(range(10), range(10))):
        raise ValueError("require exactly the predetermined 100 Long development episodes")
    policies = {r["policy_id"] for _, r in selected.values()}
    if len(policies) != 1:
        raise ValueError("analysis must use one selected SFT checkpoint")
    observations, anchors = [], []
    for (task, init), (path, row) in sorted(selected.items()):
        if not row.get("trajectory") or not row.get("decisions_path"):
            raise ValueError(f"missing diagnostic trajectory: {row['episode_uid']}")
        decisions = json.loads(Path(row["decisions_path"]).read_text())["decisions"]
        decisions = [d for d in decisions if d["end_step"] > d["start_step"]]
        if len(decisions) < 2:
            raise ValueError("too few decisions; report missing diagnostic coverage instead of replacing the episode")
        for quantile in (0.25, 0.75):
            d = decisions[round(quantile * (len(decisions)-1))]
            observations.append({"result_path": str(path), "step": d["start_step"],
                                 "task_id": task, "observation_id": f"t{task}-i{init}-q{quantile}"})
        if init < 4:
            quantile = (init+1)/5
            d = decisions[round(quantile * (len(decisions)-1))]
            anchors.append({"result_path": str(path), "step": d["start_step"],
                            "task_id": task, "anchor_id": f"t{task}-i{init}", "time_quantile": quantile})
    plan = {"purpose": "research", "namespace": namespace, "training_eligible": False,
            "continuation_policy_id": next(iter(policies)), "observations": observations, "anchors": anchors,
            "condition_samples": 2400, "branch_continuations": 480,
            "source_episodes": [r["episode_uid"] for _, r in selected.values()]}
    write_json(output, plan)


def labels(datasets, output):
    """Compare aligned labels; candidate agreement is not calibrated confidence."""
    from scipy.stats import spearmanr

    arrays, manifests = [], []
    for path in map(Path, datasets):
        manifests.append(json.loads((path / "e0-manifest.json").read_text()))
        with np.load(path / "advantage-diagnostics.npz") as data:
            arrays.append({key: data[key] for key in data.files})
    if len(arrays) < 2:
        raise ValueError("supply at least two critic-labeled snapshots")
    identity = [(e["episode_uid"], e["length"]) for e in manifests[0]["episodes"]]
    for manifest, data in zip(manifests[1:], arrays[1:]):
        if [(e["episode_uid"], e["length"]) for e in manifest["episodes"]] != identity:
            raise ValueError("critics must be compared on the identical ordered frame pool")
        for key in ("task_index", "expert"):
            np.testing.assert_array_equal(data[key], arrays[0][key])
    task_ids, expert = arrays[0]["task_index"], arrays[0]["expert"].astype(bool)
    progress = np.concatenate([np.arange(e["length"]) / max(1, e["length"]-1)
                               for e in manifests[0]["episodes"]])
    phase = np.minimum((progress * 4).astype(int), 3)
    label_arrays = [condition_labels(task_ids, a["advantage"], expert)[0] for a in arrays]
    pairs = []
    for left, right in itertools.combinations(range(len(arrays)), 2):
        per_task = {}
        for task in np.unique(task_ids[~expert]):
            mask = (task_ids == task) & ~expert
            a, b = arrays[left]["advantage"][mask], arrays[right]["advantage"][mask]
            low, high = np.percentile(a, [65, 75])
            boundary = (a >= low) & (a <= high)
            different = label_arrays[left][mask] != label_arrays[right][mask]
            correlation = float(spearmanr(a, b).statistic)
            per_task[str(task)] = {"frames": int(mask.sum()),
                "spearman": correlation if np.isfinite(correlation) else None,
                "label_flip_fraction": float(different.mean()),
                "near_threshold_flip_fraction": float(different[boundary].mean()) if boundary.any() else None,
                "left_positive_fraction": float((label_arrays[left][mask] == "positive").mean()),
                "right_positive_fraction": float((label_arrays[right][mask] == "positive").mean())}
            phase_rows = []
            for quarter in range(4):
                part = phase[mask] == quarter
                if part.any():
                    phase_rows.append({"trajectory_quarter": quarter, "frames": int(part.sum()),
                                       "label_flip_fraction": float(different[part].mean()),
                                       "left_advantage_mean": float(a[part].mean()),
                                       "right_advantage_mean": float(b[part].mean())})
            per_task[str(task)]["trajectory_progress"] = phase_rows
        pairs.append({"left": datasets[left], "right": datasets[right], "tasks": per_task})
    write_json(output, {"purpose": "research", "training_eligible": False,
                        "interpretation": "candidate consistency, not calibrated confidence", "pairs": pairs})


def conditions(manifest, checkpoint, plan, output):
    from openpi.e0.config import actor_config, read_manifest
    from openpi.policies.policy_config import create_trained_policy
    from openpi.shared.normalize import load
    from openpi_client.e0.research import policy_request

    spec = read_manifest(manifest)
    config = actor_config(spec, inference=True)
    ckpt = json.loads((Path(checkpoint) / "e0-checkpoint.json").read_text())
    if ckpt["manifest_id"] != spec["manifest_id"]:
        raise ValueError("checkpoint mismatch")
    plan = json.loads(Path(plan).read_text())
    if plan["purpose"] != "research" or not plan["namespace"].startswith("research/"):
        raise ValueError("research plan required")
    policy = create_trained_policy(config, checkpoint, norm_stats=load(Path(spec["norm_assets"]) / "libero"),
                                   sample_kwargs={"num_steps": 10, "adv_guidance_beta": 1.0})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    actions, diagnostics = [], []
    for item in plan["observations"]:
        row = json.loads(Path(item["result_path"]).read_text())
        if row["purpose"] != "development":
            raise ValueError("only development trajectories may be analyzed")
        with np.load(row["trajectory"]) as data:
            arrays = {key: data[key][item["step"]] for key in ("image", "wrist_image", "state")}
        generated = []
        for noise in range(4):
            seed = stable_seed(plan["namespace"], item["observation_id"], noise)
            generated.append([policy.infer(policy_request(arrays, row["prompt"], seed, c))["actions"]
                              for c in ("positive", "negative", "unconditional")])
        generated = np.asarray(generated)
        if generated.shape != (4, 3, 10, 7) or not np.isfinite(generated).all():
            raise ValueError("invalid condition response")
        actions.append(generated)
        prefix = generated[:, :, :5, :7]
        condition_response = float(np.mean((prefix[:, 0] - prefix[:, 1])**2))
        noise_response = float(np.mean([(prefix[i, 0]-prefix[j, 0])**2 for i, j in itertools.combinations(range(4), 2)]))
        diagnostics.append(dict(item, condition_mse=condition_response, noise_mse=noise_response,
                                condition_mse_by_action_dim=np.mean((prefix[:, 0]-prefix[:, 1])**2, axis=(0, 1)).tolist(),
                                ratio=condition_response/noise_response if noise_response else None))
    np.savez_compressed(output / "actions.npz", actions=np.asarray(actions))
    write_json(output / "report.json", {"purpose": "research", "training_eligible": False,
                "plan_id": content_id(plan), "checkpoint": str(checkpoint), "observations": diagnostics,
                "dimensions": "4 noises x [positive, negative, unconditional] x H10 x 7; metrics use K5",
                "loss_diagnostic": "separate labeled-pool loss probe required; SFT dev frames lack R1 training labels"})


def loss_probe(manifest, checkpoint, output, frames_per_task=32):
    """Paired flow-loss audit on a fixed R1/R2 training pool; not a generalization score."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.e0.config import actor_config, read_manifest
    from openpi.e0.sampling import collate
    from openpi.models import model as model_api
    from openpi.training import data_loader as dl

    spec = read_manifest(manifest)
    if spec["kind"] != "actor" or spec["stage"] not in ("r1", "r2") or frames_per_task < 8 or frames_per_task % 8:
        raise ValueError("requires RECAP actor and frames_per_task a positive multiple of 8")
    checkpoint_record = json.loads((Path(checkpoint) / "e0-checkpoint.json").read_text())
    if checkpoint_record["manifest_id"] != spec["manifest_id"]:
        raise ValueError("checkpoint mismatch")
    config = actor_config(spec, inference=True)
    data_config = config.data.create(config.assets_dirs, config.model)
    raw = dl.create_torch_dataset(data_config, 10, config.model)
    model = config.model.load(model_api.restore_params(Path(checkpoint) / "params", dtype=jnp.bfloat16))
    model.eval()
    graph, params = nnx.split(model)

    @jax.jit
    def evaluate(params, rng, batch):
        model = nnx.merge(graph, params)
        return model.compute_loss(rng, model_api.Observation.from_dict(batch), batch["actions"], train=False)

    data_manifest = json.loads(Path(spec["dataset_manifest"]).read_text())
    task_frames, offset = {}, 0
    for episode in data_manifest["episodes"]:
        if episode["source"] == "rollout":
            task_frames.setdefault(episode["task_index"], []).extend(range(offset, offset + episode["length"]))
        offset += episode["length"]
    if set(task_frames) != set(data_manifest["long_task_indices"]):
        raise ValueError("loss audit requires autonomous frames from all ten Long tasks")
    rows = []
    for task, frames in sorted(task_frames.items()):
        rng = np.random.default_rng(stable_seed("research/loss", spec["seed"], task))
        indices = rng.choice(frames, size=frames_per_task, replace=False).tolist()
        items = [raw[i] for i in indices]
        permutation = rng.permutation(len(items))
        labels = [item["adv_ind"] for item in items]
        altered = [dict(item, adv_ind=labels[j]) for item, j in zip(items, permutation)]
        correct = dl.transform_dataset(items, data_config)
        shuffled = dl.transform_dataset(altered, data_config)
        paired = []
        for begin in range(0, len(items), 8):
            key = jax.random.key(stable_seed("research/loss/noise", task, begin))
            a = collate([correct[i] for i in range(begin, begin+8)])
            b = collate([shuffled[i] for i in range(begin, begin+8)])
            left, right = float(evaluate(params, key, a)), float(evaluate(params, key, b))
            if not np.isfinite([left, right]).all():
                raise FloatingPointError("nonfinite loss diagnostic")
            paired.append({"correct": left, "shuffled": right, "difference": right-left})
        rows.append({"task": task, "frame_indices": indices,
                     "permutation": permutation.tolist(), "batches": paired,
                     "changed_fraction": float(np.mean([labels[i] != labels[j] for i, j in enumerate(permutation)]))})
    write_json(output, {"purpose": "research", "training_eligible": False,
                        "checkpoint": str(checkpoint), "dataset_id": data_manifest["dataset_id"],
                        "interpretation": "in-pool condition utilization, not held-out performance",
                        "loss": "original H10 flow loss, including original time weighting and padding",
                        "tasks": rows})
