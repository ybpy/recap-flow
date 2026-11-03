"""Run the seed-42 critic study on the development node; restart completed stages by identity."""

import collections
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np
from openpi_client.e0.artifacts import content_id, read_jsonl, write_json, write_jsonl
from openpi_client.e0.protocol import episodes, summarize_attempts, validate_grid

from openpi.e0.reward_study import (
    CLIP_METHODS,
    METHODS,
    MODES,
    fit_scales,
    load_dataset,
    load_labels,
    save_labels,
    targets,
)

RT = Path("/mnt/recap-flow-runtime")
P2 = RT / "evaluation/pilot-s42-sft-pro5000"
SFT = RT / "jobs/pilot-s42-sft-50k-pro5000.json"
DEMOS = RT / "datasets/sft-v1"
ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def cmd(*parts, gpu=None):
    env = dict(os.environ, E0_PLATFORM="pro5000", CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" if gpu is None else str(gpu))
    subprocess.run(["bash", "scripts/e0.sh", *map(str, parts)], cwd=ROOT, env=env, check=True)


def stage_dir(dest, marker, fn):
    if (dest / marker).is_file():
        return
    tmp = dest.with_name(dest.name + ".partial-" + uuid.uuid4().hex[:8])
    fn(tmp)
    if not (tmp / marker).is_file():
        raise RuntimeError(f"stage did not produce {marker}: {tmp}")
    tmp.rename(dest)


def verify_attempts(manifest, results, purpose, policy):
    rows = [read(p) for p in sorted((results / "attempts").glob("*.json"))]
    if not summarize_attempts(read_jsonl(manifest), rows)["complete"]:
        raise ValueError(f"incomplete {purpose} attempts")
    rows = [row for row in rows if row["suite"] == "libero_10"]
    validate_grid(rows, ["libero_10"], [0], purpose)
    if {r["policy_id"] for r in rows} != {policy}:
        raise ValueError("mixed behavior checkpoints")
    return rows


def difference_report(pool, scales):
    report = {}
    for ep in pool["episodes"]:
        task, length = ep["task_index"], ep["length"]
        success = True if ep["source"] == "demo" else ep["success"]
        b, c = [targets(m, length, success, task, scales) for m in METHODS[1:]]
        counts = report.setdefault(str(task), collections.Counter())
        counts.update(
            episodes=1,
            frames=length,
            demo=int(ep["source"] == "demo"),
            rollout=int(ep["source"] == "rollout"),
            success=int(success),
            timeout=int(not success and length == 520),
            value_different=int(np.sum(b[:, 1] != c[:, 1])),
            reward_different=int(np.sum(b[:, 0] != c[:, 0])),
        )
    return dict(scales=scales, by_task=report)


def run(output, wait=False, clip_from=None):
    work = Path(output).resolve()
    parent = Path(clip_from).resolve() if clip_from else None
    if parent and (work == parent or parent in work.parents or work in parent.parents):
        raise ValueError("clip supplement needs a separate sibling study directory")
    work.mkdir(parents=True, exist_ok=True)
    with (work / "study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def status(phase, **info):
            write_json(work / "status.json", dict(phase=phase, unix=time.time(), **info))

        try:
            if parent:
                if read(parent / "status.json")["phase"] != "complete":
                    raise ValueError("the original study must be complete")
                status("verify-sealed-inputs")
                run_critics(work, parent, CLIP_METHODS, status)
                status("complete", report=str(work / "report/index.html"))
                return
            status("awaiting-sft-selection")
            while not (P2 / "selection.json").is_file():
                if not wait:
                    raise FileNotFoundError(P2 / "selection.json")
                time.sleep(60)
            pipeline(work, status)
            status("complete", report=str(work / "report/index.html"))
        except BaseException as exc:
            prior = read(work / "status.json")
            status("failed", stage=prior["phase"], error=repr(exc))
            raise


def pipeline(work, status):
    selected = read(P2 / "selection.json")
    ckpt = Path(selected["checkpoint"])
    policy = content_id(read(ckpt / "e0-checkpoint.json"))
    winner = next(row for row in selected["candidates"] if row["updates"] == selected["updates"])
    inputs = {"validation": (Path(winner["episode_manifest"]), Path(winner["results"]), "development")}
    for role, purpose in (("train", "rollout"), ("review", "review")):
        status("collect-" + role)
        manifest, out = work / (role + "-episodes.jsonl"), work / (role + "-evaluation")
        rows = list(episodes("critic-study/s42/r1/" + role, ["libero_10"], purpose=purpose, training_seed=42))
        if manifest.exists() and read_jsonl(manifest) != rows:
            raise ValueError("existing episode grid differs")
        write_jsonl(manifest, rows)
        cmd(
            "evaluate",
            "--actor-manifest",
            SFT,
            "--checkpoint",
            ckpt,
            "--episode-manifest",
            manifest,
            "--output",
            out,
            "--platform",
            "pro5000",
            "--devices",
            *range(8),
            "--workers-per-device",
            4,
        )
        inputs[role] = (manifest, out / "results", purpose)
    seen = set()
    for role, (manifest, results, purpose) in inputs.items():
        rows = verify_attempts(manifest, results, purpose, policy)
        ids = {r["episode_uid"] for r in rows}
        if seen & ids:
            raise ValueError("overlapping study splits")
        seen.update(ids)
        status("convert-" + role)
        stage_dir(
            work / role,
            "e0-manifest.json",
            lambda tmp: cmd(
                "convert-rollouts",
                "--episode-manifest",
                manifest,
                "--results-dir",
                results,
                "--demo-dir",
                DEMOS,
                "--output",
                tmp,
                "--purpose",
                purpose,
                "--branch",
                "shared",
                "--round-index",
                1,
                "--suite",
                "libero_10",
            ),
        )
    pool = work / "pool"
    stage_dir(pool, "e0-manifest.json", lambda tmp: cmd("merge", "--sources", DEMOS, work / "train", "--output", tmp))
    run_critics(work, work, METHODS, status)


def audit_clip(parent, work, data, method):
    original = method.removesuffix("_clip")
    old_spec = read(parent / f"{original}-{MODES[0]}" / "training.json")
    path = work / f"labels-{method}" / "protocol.json"
    new_spec = dict(old_spec, reward_protocol=str(path), reward_protocol_id=read(path)["protocol_id"])
    report = {}
    for role, dataset in data.items():
        old, _ = load_labels(old_spec, dataset, role)
        new, scale = load_labels(new_spec, dataset, role)
        if scale != 1:
            raise ValueError("direct-clip training and prediction require a unit head scale")
        np.testing.assert_array_equal(new[:, 0], old[:, 0])
        np.testing.assert_array_equal(new[:, 2], old[:, 1])
        np.testing.assert_array_equal(new[:, 1], np.clip(old[:, 1], -1, 0))
        raw = new[:, 2]
        clipped = raw < -1
        offset, success_frames, success_episodes = 0, 0, 0
        for ep in dataset["episodes"]:
            n = ep["length"]
            if ep["source"] == "demo" or ep["success"]:
                changed = int(clipped[offset : offset + n].sum())
                success_frames += changed
                success_episodes += int(changed > 0)
            offset += n
        report[role] = dict(
            frames=len(raw),
            below=int(clipped.sum()),
            equal=int((raw == -1).sum()),
            clipped_success_frames=success_frames,
            clipped_success_episodes=success_episodes,
            single_step_rewards_match=True,
        )
    expected_review = (8175, 76, 2) if original == "task_pool_max" else (8029, 0, 0)
    rev = report["review"]
    if (report["train"]["frames"], report["train"]["below"], report["validation"]["below"]) != (411854, 9583, 7770) or (
        rev["below"],
        rev["clipped_success_frames"],
        rev["clipped_success_episodes"],
    ) != expected_review:
        raise ValueError(f"unexpected clipping counts for the sealed seed-42 data: {report}")
    return report


def run_critics(work, source, methods, status):
    from openpi.e0.jobs import training_config
    from openpi.e0.selection import critic

    pool = source / "pool"
    data = {
        k: load_dataset(p)
        for k, p in (("train", pool), ("validation", source / "validation"), ("review", source / "review"))
    }
    scales = fit_scales(data["train"])
    differences = difference_report(data["train"], scales)
    differences["task_descriptions"] = read_jsonl(pool / "meta/tasks.jsonl")
    differences["long_sim_task_to_dataset_task"] = {
        ep["episode_uid"].split("/task-")[-1].split("/")[0]: ep["task_index"]
        for ep in data["train"]["episodes"]
        if ep["source"] == "rollout"
    }
    write_json(work / "target-differences.json", differences)
    audit = {}
    for method in methods:
        labels = work / ("labels-" + method)
        stage_dir(labels, "protocol.json", lambda tmp: save_labels(tmp, method, scales, data))
        if method in CLIP_METHODS:
            audit[method] = audit_clip(source, work, data, method)
    if audit:
        write_json(work / "clip-audit.json", audit)
    jobs = []
    for method in methods:
        labels = work / ("labels-" + method)
        protocol = read(labels / "protocol.json")
        for mode in MODES:
            name = f"{method}-{mode}"
            base = work / name
            base.mkdir(exist_ok=True)
            manifest = base / "training.json"
            request = dict(
                id=f"critic-study-s42-{name}",
                stage="r1",
                kind="critic",
                branch="shared",
                seed=42,
                data_dir=str(pool),
                num_train_steps=20000,
                checkpoint_updates=[10000, 15000, 20000],
                freeze_mode=mode,
                reward_protocol=str(labels / "protocol.json"),
                reward_protocol_id=protocol["protocol_id"],
            )
            write_json(base / "request.json", request)
            spec = training_config(base / "request.json", manifest)
            if audit:
                old = read(source / f"{method.removesuffix('_clip')}-{mode}" / "training.json")
                varying = {"id", "run_dir", "manifest_id", "reward_protocol", "reward_protocol_id"}
                if {k: v for k, v in spec.items() if k not in varying} != {
                    k: v for k, v in old.items() if k not in varying
                }:
                    raise ValueError("paired critic training manifests differ outside reward and run identity")
            run_dir = Path(spec["run_dir"])
            status("train-" + name, manifest=str(manifest))
            if not (run_dir / "status.json").is_file() or not read(run_dir / "status.json")["complete"]:
                cmd("resume" if (run_dir / "resolved-manifest.json").exists() else "run", manifest)
            paths = []
            for step in spec["checkpoint_updates"]:
                status(f"validate-{name}-{step}")
                dest = base / f"validation-{step}"
                stage_dir(
                    dest,
                    "predictions.json",
                    lambda tmp: cmd(
                        "predict",
                        manifest,
                        "--checkpoint",
                        run_dir / "candidates" / str(step),
                        "--data-dir",
                        source / "validation",
                        "--output",
                        tmp,
                        gpu=0,
                    ),
                )
                paths.append(dest)
            critic(manifest, paths, base / "selection.json")
            jobs.append((name, manifest, base, read(base / "selection.json")["checkpoint"]))
    for name, manifest, base, candidate in jobs:
        for role, directory, flag in (("train", pool, "--autonomous-only"), ("review", source / "review", "--review")):
            status(f"predict-{role}-{name}")
            stage_dir(
                base / role,
                "predictions.json",
                lambda tmp: cmd(
                    "predict",
                    manifest,
                    "--checkpoint",
                    candidate,
                    "--data-dir",
                    directory,
                    "--output",
                    tmp,
                    flag,
                    gpu=0,
                ),
            )
    status("paired-report")
    from openpi.e0.critic_report import report

    stage_dir(work / "report", "index.html", lambda tmp: report(work, tmp, parent=source if audit else None))
