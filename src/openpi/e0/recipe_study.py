"""Manual TIONE jobs for Long-only sampling experiments; no cloud submission."""

import fcntl
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
from openpi_client.e0.artifacts import content_id, read_jsonl, sha256_file, write_json, write_jsonl
from openpi_client.e0.protocol import episodes, summarize_attempts, validate_grid

from openpi.e0.actor_study import seal
from openpi.e0.config import read_manifest
from openpi.e0.jobs import ROOT, RUNTIME, training_config
from openpi.e0.sampling import FrameStream, RECIPES

BASE = RUNTIME / "actor-study/s42-bc-clip-v1"
DEFAULT = RUNTIME / "actor-study/s42-long-recipes-v1"
VARIANTS = {"long-pooled": "long_pooled_v1", "long-demo70": "long_demo70_rollout30_v1"}


def read(path):
    return json.loads(Path(path).read_text())


def sampling_audit(dataset, draws=100000):
    manifest = read(dataset)
    eps = manifest["episodes"]
    ends = np.cumsum([e["length"] for e in eps])
    tasks = np.asarray([e["task_index"] for e in eps])
    rollout = np.asarray([e["source"] == "rollout" for e in eps])
    long = set(manifest["long_task_indices"])
    rows = {}
    for name in ("e0_75_25_pooled_v1", *VARIANTS.values()):
        recipe = RECIPES[name]
        stream = FrameStream.from_manifest(dataset, 42, "recap", recipe=recipe)
        indices = np.searchsorted(ends, [stream.frame(d) for d in range(draws)], side="right")
        by_task = {}
        expected_rollout = 0.0
        for task in sorted(set(tasks)):
            mask = tasks == task
            demo = sum(e["length"] for e in eps if e["task_index"] == task and e["source"] == "demo")
            autonomous = sum(e["length"] for e in eps if e["task_index"] == task and e["source"] == "rollout")
            mass = recipe["long_probability"] / 10 if task in long else (1 - recipe["long_probability"]) / 30
            fraction = 0.3 if recipe["source_sampling"] == "demo_rollout" and task in long else autonomous / (demo + autonomous)
            expected_rollout += mass * fraction
            actual = tasks[indices] == task
            by_task[str(task)] = dict(demo_frames=demo, rollout_frames=autonomous,
                                     expected_probability=mass, expected_rollout_fraction=fraction,
                                     sampled_frames=int(actual.sum()),
                                     sampled_rollout_frames=int((actual & rollout[indices]).sum()))
        rows[name] = dict(draws=draws, expected_demo_fraction=1 - expected_rollout,
                          expected_rollout_fraction=expected_rollout,
                          sampled_rollout_fraction=float(rollout[indices].mean()), by_task=by_task)
    return dict(dataset_id=manifest["dataset_id"], recipes=rows)


def prepare(output=DEFAULT):
    work = Path(output)
    work.mkdir(parents=True, exist_ok=True)
    baseline = read_manifest(BASE / "B/training.json")
    state = read(Path(baseline["run_dir"]) / "status.json")
    if not state["complete"] or state["completed_updates"] != 60000:
        raise ValueError("the existing B baseline must be a completed 60k run")
    dataset = read(baseline["dataset_manifest"])
    if len(dataset["episodes"]) != 2193 or sum(e["length"] for e in dataset["episodes"]) != 411854:
        raise ValueError("unexpected frozen R1 pool")
    source_files = [BASE / "B/training.json", BASE / "B/selection.json", BASE / "B/evaluations.json",
                    BASE / "comparison.json", BASE / "development-episodes.jsonl",
                    Path(baseline["dataset_manifest"]), Path(baseline["data_dir"]) / "advantage-diagnostics.npz"]
    comparison = dict(
        study="s42-long-recipes-v1", seed=42, training_repeats=1, branch="base_reset",
        baseline="B-clip/all_backbones/20k EMA labels", data_dir=baseline["data_dir"],
        dataset_id=baseline["dataset_id"], norm_stats_sha256=baseline["norm_stats_sha256"],
        source_files={str(p): sha256_file(p) for p in source_files},
        updates=60000, candidates=[30000, 45000, 60000], variants=VARIANTS,
        sft_selection=read(BASE / "comparison.json")["sft_selection"],
        baseline_selection=read(BASE / "B/selection.json"),
        interpretation="One training seed; removing other suites also increases Long exposure at fixed updates.",
    )
    seal(work / "comparison.json", comparison)
    development = read_jsonl(BASE / "development-episodes.jsonl")
    confirmation = list(episodes("actor-study/s42-long-recipes-v1/confirmation", ["libero_10"],
                                 groups=[0, 1, 2], purpose="review", training_seed=42))
    validate_grid(development, ["libero_10"], [0], "development")
    validate_grid(confirmation, ["libero_10"], [0, 1, 2], "review")
    for phase, rows in (("development", development), ("confirmation", confirmation)):
        path = work / (phase + "-episodes.jsonl")
        if path.exists() and read_jsonl(path) != rows:
            raise ValueError("sealed evaluation manifest changed")
        write_jsonl(path, rows)
    for variant, recipe in VARIANTS.items():
        request = dict(id="recipe-s42-" + variant, kind="actor", stage="r1", branch="base_reset", seed=42,
                       platform="pro5000", data_dir=baseline["data_dir"], num_train_steps=60000,
                       checkpoint_updates=[30000, 45000, 60000], sampling_recipe=RECIPES[recipe])
        seal(work / variant / "request.json", request)
        training_config(work / variant / "request.json", work / variant / "training.json")
        for count in (2, 8):
            smoke = dict(request, id=f"smoke-recipe-s42-{variant}-{count}gpu-v1", smoke=True,
                         num_train_steps=20, warmup_steps=10, batch_size=16 if count == 2 else 256,
                         fsdp_devices=2 if count == 2 else 4, num_workers=2 if count == 2 else 8,
                         checkpoint_interval=10, checkpoint_updates=[20])
            if count == 2:
                smoke["debug_gpu_count"] = 2
            seal(work / variant / f"smoke-{count}gpu-request.json", smoke)
            training_config(work / variant / f"smoke-{count}gpu-request.json", work / variant / f"smoke-{count}gpu.json")
    audit = work / "sampling-audit.json"
    if not audit.exists():
        write_json(audit, sampling_audit(baseline["dataset_manifest"]))
    write_json(work / "preparation.json", dict(complete=True, new_training_runs=2, updates_per_run=60000,
                                              new_development_attempts=3000, confirmation_attempts=6000))


def check_inputs(work):
    comparison = read(work / "comparison.json")
    for path, identity in comparison["source_files"].items():
        if sha256_file(path) != identity:
            raise ValueError(f"frozen comparison input changed: {path}")
    return comparison


def _command(args, log, timeout=604800):
    from openpi.e0.smoke import command
    command(args, log, dict(os.environ, E0_PLATFORM="pro5000"), timeout=timeout)


def acceptance(work, variant, gpu_count):
    """A recipe-specific full-state recovery test; eight GPUs are required for launch."""
    from openpi.e0.smoke import train, verify_recovery
    from openpi.e0.provenance import code_record, runtime_identity

    root = work / variant / f"acceptance-{gpu_count}gpu"
    root.mkdir(parents=True, exist_ok=True)
    seal(root / "execution.json", dict(source_id=code_record(ROOT)["source_id"],
                                       runtime_id=runtime_identity()["runtime_id"]))
    env = dict(os.environ, E0_PLATFORM="pro5000")
    manifest = work / variant / f"smoke-{gpu_count}gpu.json"
    train(manifest, env, root / "train.log")
    verify_recovery(manifest, root / "recovery.json", env, root / "recovery.log")
    write_json(root / "result.json", dict(complete=True, smoke=True, gpu_count=gpu_count,
                                          full_state_bitwise_equal=True,
                                          batch=16 if gpu_count == 2 else 256,
                                          fsdp=2 if gpu_count == 2 else 4))


def task(output=DEFAULT, variant=None, phase="train-dev", gpu_count=8):
    work = Path(output)
    if variant not in VARIANTS:
        raise ValueError("a declared recipe variant is required")
    if phase not in ("smoke", "train-dev", "evaluate") or gpu_count not in (2, 8):
        raise ValueError("invalid task phase/hardware")
    if phase != "smoke" and gpu_count != 8:
        raise ValueError("pilot training/evaluation tasks require eight GPUs")
    check_inputs(work)
    path = work / variant
    path.mkdir(parents=True, exist_ok=True)
    with (path / "task.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from openpi.e0.provenance import code_record, runtime_identity
        seal(path / f"execution-{gpu_count}gpu.json", dict(source_id=code_record(ROOT)["source_id"],
                                                         runtime_id=runtime_identity()["runtime_id"]))
        def status(stage, **extra):
            write_json(path / f"task-status-{gpu_count}gpu.json", dict(phase=stage, unix=time.time(), **extra))
        try:
            if phase != "evaluate":
                status("preflight")
                args = ["preflight", "--output", path / f"preflight-{gpu_count}gpu.json"]
                if gpu_count == 2:
                    args += ["--debug-gpu-count", 2]
                _command(args, path / f"preflight-{gpu_count}gpu.log", timeout=1800)
                status("acceptance")
                acceptance(work, variant, gpu_count)
                if phase == "smoke":
                    status("smoke-complete")
                    return
                status("train")
                from openpi.e0.actor_study import train
                train(path / "training.json")
            status("development")
            from openpi.e0.evaluation import evaluate
            from openpi.e0.selection import actor
            manifest = path / "training.json"
            spec = read_manifest(manifest)
            if not read(Path(spec["run_dir"]) / "status.json")["complete"]:
                raise ValueError("development selection requires the completed 60k run")
            entries = []
            for update in spec["checkpoint_updates"]:
                destination = path / "development" / str(update)
                evaluate(manifest, Path(spec["run_dir"]) / "candidates" / str(update),
                         work / "development-episodes.jsonl", destination,
                         devices=range(8), workers_per_device=4, platform="pro5000")
                entries.append(dict(updates=update, episode_manifest=str(work / "development-episodes.jsonl"),
                                    results=str(destination / "results")))
            seal(path / "evaluations.json", entries)
            actor(manifest, path / "evaluations.json", path / "selection-check.json")
            seal(path / "selection.json", read(path / "selection-check.json"))
            status("complete", selection=read(path / "selection.json"))
        except BaseException as exc:
            previous = read(path / f"task-status-{gpu_count}gpu.json")
            status("failed", stage=previous["phase"], error=repr(exc))
            raise


def selected_inputs(work):
    from openpi.e0.critic_study import SFT
    from openpi.e0.selection import actor
    comparison = check_inputs(work)
    chosen = {}
    for name in VARIANTS:
        manifest = work / name / "training.json"
        spec = read_manifest(manifest)
        if not read(Path(spec["run_dir"]) / "status.json")["complete"]:
            raise ValueError("both 60k recipe runs must be complete")
        actor(manifest, work / name / "evaluations.json", work / name / "selection-check.json")
        if read(work / name / "selection-check.json") != read(work / name / "selection.json"):
            raise ValueError("sealed selection changed")
        chosen[name] = dict(manifest=str(manifest), selection=read(work / name / "selection.json"))
    chosen["B-reference"] = dict(manifest=str(BASE / "B/training.json"), selection=comparison["baseline_selection"])
    chosen["SFT"] = dict(manifest=str(SFT), selection=comparison["sft_selection"])
    for value in chosen.values():
        spec = read_manifest(value["manifest"])
        meta = read(Path(value["selection"]["checkpoint"]) / "e0-checkpoint.json")
        if meta["manifest_id"] != spec["manifest_id"] or meta["parameter_kind"] != "EMA" or meta["smoke"]:
            raise ValueError("selected checkpoint binding differs")
    return chosen


def confirm(output=DEFAULT):
    from openpi.e0.evaluation import evaluate
    from openpi.e0.provenance import code_record, runtime_identity
    work = Path(output)
    with (work / "confirmation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        chosen = selected_inputs(work)
        seal(work / "confirmation-inputs.json", chosen)
        seal(work / "confirmation-execution.json", dict(source_id=code_record(ROOT)["source_id"],
                                                        runtime_id=runtime_identity()["runtime_id"]))
        _command(["preflight", "--output", work / "confirmation-preflight.json"], work / "confirmation-preflight.log", 1800)
        for name, value in chosen.items():
            write_json(work / "confirmation-status.json", dict(phase=name, unix=time.time()))
            evaluate(value["manifest"], value["selection"]["checkpoint"], work / "confirmation-episodes.jsonl",
                     work / name / "confirmation", devices=range(8), workers_per_device=4, platform="pro5000")
        report(work)
        write_json(work / "confirmation-status.json", dict(phase="complete", unix=time.time()))


def report(work):
    from openpi.e0.actor_report import interval
    work = Path(work)
    chosen = read(work / "confirmation-inputs.json")
    expected = read_jsonl(work / "confirmation-episodes.jsonl")
    validate_grid(expected, ["libero_10"], [0, 1, 2], "review")
    models, tensors = {}, {}
    for name, value in chosen.items():
        root = work / name / "confirmation/results"
        policy = content_id(read(Path(value["selection"]["checkpoint"]) / "e0-checkpoint.json"))
        rows = [read(p) for p in sorted((root / "attempts").glob("*.json"))]
        summary = summarize_attempts(expected, rows)
        if not summary["complete"] or read(root / "protocol.json")["policy_id"] != policy or any(r["policy_id"] != policy for r in rows):
            raise ValueError("confirmation denominator or policy identity differs")
        values = np.zeros((10, 50, 3))
        for row in rows:
            values[row["task_id"], row["initial_state_id"], row["group"]] = row["success"]
        tensors[name] = values
        models[name] = dict(summary, updates=value["selection"]["updates"],
                            group_success=values.mean(axis=(0, 1)).tolist(), checkpoint=value["selection"]["checkpoint"])
    draws = np.random.default_rng(42).integers(0, 50, (10, 10000, 50))
    pairs = {a + "-" + b: interval(tensors[a] - tensors[b], draws) for a, b in itertools.combinations(chosen, 2)}
    write_json(work / "report/results.json", dict(models=models, pairs=pairs, training_repeats=1,
                                                  evaluation_groups=3, bootstrap="paired initial-state clusters within task"))
    table = ["<html><meta charset='utf-8'><title>Long-only recipe comparison</title><h1>Long-only recipe comparison</h1>",
             "<p>One training seed (42); three paired evaluation groups. Each method has 1,500 attempts.</p>",
             "<table><tr><th>Method</th><th>Updates</th><th>Success</th></tr>"]
    for name, values in models.items():
        table.append(f"<tr><td>{name}</td><td>{values['updates']}</td><td>{100*values['macro_success']:.2f}%</td></tr>")
    table.append("</table><p>See results.json for per-task results and paired bootstrap intervals.</p></html>")
    (work / "report/index.html").write_text("\n".join(table))
