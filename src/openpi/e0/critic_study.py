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

from openpi.e0.reward_study import METHODS, MODES, fit_scales, load_dataset, save_labels, targets

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


def run(output, wait=False):
    work = Path(output).resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work / "study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def status(phase, **info):
            write_json(work / "status.json", dict(phase=phase, unix=time.time(), **info))

        try:
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
    from openpi.e0.jobs import training_config
    from openpi.e0.selection import critic

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
    data = {
        k: load_dataset(p)
        for k, p in (("train", pool), ("validation", work / "validation"), ("review", work / "review"))
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
    jobs = []
    for method in METHODS:
        labels = work / ("labels-" + method)
        stage_dir(labels, "protocol.json", lambda tmp: save_labels(tmp, method, scales, data))
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
                        work / "validation",
                        "--output",
                        tmp,
                        gpu=0,
                    ),
                )
                paths.append(dest)
            critic(manifest, paths, base / "selection.json")
            jobs.append((name, manifest, base, read(base / "selection.json")["checkpoint"]))
    for name, manifest, base, candidate in jobs:
        for role, directory, flag in (("train", pool, "--autonomous-only"), ("review", work / "review", "--review")):
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

    stage_dir(work / "report", "index.html", lambda tmp: report(work, tmp))
