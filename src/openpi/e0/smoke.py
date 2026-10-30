"""Bounded local smoke jobs; TIONE launch remains a manual user action."""

import json
import os
from pathlib import Path
import socket
import subprocess
import time

from openpi_client.e0.artifacts import read_jsonl
from openpi_client.e0.artifacts import write_json
from openpi_client.e0.artifacts import write_jsonl

from openpi.e0.jobs import ROOT
from openpi.e0.jobs import smoke_configs


def command(args, log, env, timeout=14400):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open("a") as out:
        out.write("\n" + json.dumps(list(map(str, args))) + "\n")
        out.flush()
        subprocess.run(
            ["bash", str(ROOT / "scripts/e0.sh"), *map(str, args)],
            cwd=ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=True,
        )


def completed(path):
    status = Path(path) / "status.json"
    return status.exists() and json.loads(status.read_text())["complete"]


def train(path, env, log):
    spec = json.loads(Path(path).read_text())
    if completed(spec["run_dir"]):
        return
    recovery = Path(spec["run_dir"]) / "recovery"
    resume = any(p.name.isdigit() for p in recovery.iterdir()) if recovery.exists() else False
    command(["resume" if resume else "run", path], log, env)


def verify_recovery(control, path, env, log):
    reference = json.loads(Path(control).read_text())
    recovered = dict(reference, id=reference["id"] + "-recovery", run_dir=reference["run_dir"] + "-recovery")
    write_json(path, recovered)
    if not completed(recovered["run_dir"]):
        run = Path(recovered["run_dir"])
        if not (run / "resolved-manifest.json").exists():
            command(["run", path, "--stop-after", "10"], log, env)
        partial = json.loads((run / "state-fingerprint.json").read_text())
        write_json(run / "expected-state-10.json", partial)
        command(["resume", path], log, env)
    a, b = Path(reference["run_dir"]), Path(recovered["run_dir"])
    if json.loads((a / "initial-state-fingerprint.json").read_text()) != json.loads(
        (b / "initial-state-fingerprint.json").read_text()
    ):
        raise AssertionError("same seed did not reproduce the full initial state")
    if json.loads((b / "expected-state-10.json").read_text()) != json.loads((b / "resumed-state-10.json").read_text()):
        raise AssertionError("restored optimizer/params/EMA/step differs from the saved state")
    left, right = read_jsonl(a / "metrics.jsonl"), read_jsonl(b / "metrics.jsonl")
    if len(left) != reference["num_train_steps"] or len(right) != reference["num_train_steps"]:
        raise AssertionError("recovery did not complete planned updates")
    for x, y in zip(left, right, strict=True):
        ignored = {"elapsed_seconds", "session_id"}
        if {k: v for k, v in x.items() if k not in ignored} != {k: v for k, v in y.items() if k not in ignored}:
            raise AssertionError(f"recovery differs at update {x['updates']}; inspect batch and numerical metrics")
    if json.loads((a / "state-fingerprint.json").read_text()) != json.loads((b / "state-fingerprint.json").read_text()):
        raise AssertionError("full optimizer/params/EMA/step state differs after recovery")
    write_json(
        Path(path).with_suffix(".verified.json"),
        {
            "complete": True,
            "updates": reference["num_train_steps"],
            "interrupted_after": 10,
            "full_state_bitwise_equal": True,
            "batches_and_metrics_equal": True,
            "initial_state_equal": True,
            "restored_state_equal": True,
            "rng_keys_equal": True,
        },
    )


def collect(actor_path, checkpoint, namespace, output, env, log):
    """Own the policy process and stop exactly that process on exit."""
    from openpi_client.e0.protocol import episodes

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    episode_path = output / "episodes.jsonl"
    write_jsonl(episode_path, episodes(namespace, ["libero_10"], attempts=2))
    from openpi_client.e0.protocol import summarize_attempts

    attempts = sorted((output / "results/attempts").glob("*.json"))
    existing = summarize_attempts(read_jsonl(episode_path), [json.loads(p.read_text()) for p in attempts])
    if existing["complete"]:
        write_json(output / "summary.json", existing)
        return episode_path, output / "results"
    server_log = (output / "server.log").open("a")
    server_env = dict(env, CUDA_VISIBLE_DEVICES="0")
    server = subprocess.Popen(
        [
            "bash",
            str(ROOT / "scripts/e0.sh"),
            "serve",
            str(actor_path),
            "--checkpoint",
            str(checkpoint),
            "--port",
            "8765",
        ],
        cwd=ROOT,
        env=server_env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 1800
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"policy server exited: {output / 'server.log'}")
            try:
                with socket.create_connection(("127.0.0.1", 8765), timeout=1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("policy warmup exceeded 30 minutes") from None
                time.sleep(2)
        workers = []
        handles = []
        try:
            for shard in range(4):
                handle = (output / f"worker-{shard}.log").open("a")
                handles.append(handle)
                worker_env = dict(env, MUJOCO_EGL_DEVICE_ID=str(shard))
                workers.append(
                    subprocess.Popen(
                        [
                            "bash",
                            str(ROOT / "scripts/e0.sh"),
                            "rollout",
                            str(episode_path),
                            "--output",
                            str(output / "results"),
                            "--port",
                            "8765",
                            "--shard",
                            str(shard),
                            "--shards",
                            "4",
                        ],
                        cwd=ROOT,
                        env=worker_env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                )
            deadline = time.monotonic() + 14400
            while any(p.poll() is None for p in workers):
                if server.poll() is not None:
                    raise RuntimeError("policy server failed during collection")
                if any(p.poll() not in (None, 0) for p in workers):
                    raise RuntimeError("rollout worker failed")
                if time.monotonic() > deadline:
                    raise TimeoutError("smoke collection exceeded four hours")
                time.sleep(2)
            if any(p.returncode for p in workers):
                raise RuntimeError("rollout failed")
        finally:
            for p in workers:
                if p.poll() is None:
                    p.terminate()
            for p in workers:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            for h in handles:
                h.close()
        command(
            [
                "report",
                "--manifest",
                episode_path,
                "--results",
                output / "results",
                "--output",
                output / "summary.json",
            ],
            log,
            env,
        )
    finally:
        if server.poll() is None:
            server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        server_log.close()
    return episode_path, output / "results"


def collect_data(paths, env, output):
    output = Path(output)
    log = output / "closed-loop.log"
    actor_path = Path(paths[0])
    actor = json.loads(actor_path.read_text())
    initial = Path(actor["run_dir"]) / "candidates/20"
    datasets = []
    for purpose in ("rollout", "development"):
        destination = output / f"{purpose}-data"
        if not (destination / "e0-manifest.json").exists():
            episodes, results = collect(
                actor_path, initial, f"smoke/{output.name}/{purpose}", output / purpose, env, log
            )
            command(
                [
                    "convert-rollouts",
                    "--episode-manifest",
                    episodes,
                    "--results-dir",
                    results,
                    "--demo-dir",
                    actor["data_dir"],
                    "--output",
                    destination,
                    "--purpose",
                    purpose,
                    "--branch",
                    "shared",
                    "--round-index",
                    "1",
                ],
                log,
                env,
            )
        datasets.append(destination)
    pool = output / "pool"
    if not (pool / "e0-manifest.json").exists():
        command(["merge", "--sources", actor["data_dir"], datasets[0], "--output", pool], log, env)
    return actor, pool, datasets


def closed_loop(paths, env, output):
    output = Path(output)
    log = output / "closed-loop.log"
    actor, pool, datasets = collect_data(paths, env, output)
    command(["replay-check", "--results", output / "rollout/results", "--output", output / "replay-passed.json"],
            log, dict(env, MUJOCO_EGL_DEVICE_ID="0"))
    critic = json.loads(Path(paths[1]).read_text())
    critic.update(
        id=critic["id"] + "-loop",
        run_dir=critic["run_dir"] + "-loop",
        data_dir=str(pool),
        dataset_manifest=str(pool / "e0-manifest.json"),
    )
    critic_path = output / "loop-critic.json"
    write_json(critic_path, critic)
    train(critic_path, env, log)
    checkpoint = Path(critic["run_dir"]) / "candidates/20"
    for data, folder, extra in ((datasets[1], "validation", []), (pool, "prediction", ["--autonomous-only"])):
        target = output / folder
        if not (target / "predictions.json").exists():
            command(
                [
                    "predict",
                    critic_path,
                    "--checkpoint",
                    checkpoint,
                    "--data-dir",
                    data,
                    "--output",
                    target,
                    "--batch-size",
                    "8",
                    *extra,
                ],
                log,
                dict(env, CUDA_VISIBLE_DEVICES="0"),
            )
    labeled = output / "labeled"
    if not (labeled / "e0-manifest.json").exists():
        command(["label", "--source", pool, "--predictions", output / "prediction", "--output", labeled], log, env)
    actor.update(
        num_train_steps=20, checkpoint_updates=[20],
        id=actor["id"] + "-r1-loop",
        run_dir=actor["run_dir"] + "-r1-loop",
        stage="r1",
        branch="base_reset",
        data_dir=str(labeled),
        dataset_manifest=str(labeled / "e0-manifest.json"),
    )
    actor_path = output / "loop-actor.json"
    write_json(actor_path, actor)
    train(actor_path, env, log)
    collect(
        actor_path,
        Path(actor["run_dir"]) / "candidates/20",
        f"smoke/{output.name}/post-update",
        output / "post-update",
        env,
        log,
    )
    write_json(
        output / "closed-loop-status.json",
        {
            "complete": True,
            "smoke": True,
            "attempts_per_task": 2,
            "training_rollout_attempts": 20,
            "independent_development_attempts": 20,
            "post_update_attempts": 20,
        },
    )


def smoke(platform, output, phase="all", jobs_index=None):
    if platform not in ("cyy_dev", "a800", "pro5000"):
        raise ValueError(platform)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, E0_PLATFORM=platform)
    name = output.name
    paths = json.loads(Path(jobs_index).read_text()) if jobs_index else smoke_configs(output / "jobs", platform, name)
    if len(paths) != 3 or not all(json.loads(Path(p).read_text())["smoke"] for p in paths):
        raise ValueError("smoke requires exactly one actor and two critic smoke manifests")
    phases = ("preflight", "training", "recovery", "closed-loop") if phase == "all" else (phase,)
    for part in phases:
        log = output / (part + ".log")
        if part == "preflight":
            command(["preflight", "--output", output / "train-preflight.json"], log, env)
            for gpu in range(4 if platform == "cyy_dev" else 8):
                command(["egl", "--output", output / f"egl-{gpu}.json"], log, dict(env, MUJOCO_EGL_DEVICE_ID=str(gpu)))
            command(["schema", "--output", output / "schema-train.json"], log, env)
            command(["schema-libero", "--output", output / "schema-libero.json"], log, env)
            command(["test"], log, env)
        elif part == "training":
            for path in paths:
                train(path, env, log)
        elif part == "recovery":
            for path in (paths[0], paths[2]):
                train(path, env, log)
                verify_recovery(path, output / (Path(path).stem + "-recovery.json"), env, log)
        elif part == "collect":
            if not completed(json.loads(Path(paths[0]).read_text())["run_dir"]):
                raise RuntimeError("actor smoke is incomplete")
            collect_data(paths, env, output)
        elif part == "closed-loop":
            for path in paths:
                if not completed(json.loads(Path(path).read_text())["run_dir"]):
                    raise RuntimeError("complete training smoke before closed loop")
            closed_loop(paths, env, output)
        else:
            raise ValueError(part)
        write_json(
            output / f"{part}-passed.json", {"complete": True, "phase": part, "platform": platform, "smoke": True}
        )
