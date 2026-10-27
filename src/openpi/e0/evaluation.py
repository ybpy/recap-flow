"""Run one policy server per GPU and paired, manifest-defined simulation workers."""

import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import uuid

from openpi_client.e0.artifacts import content_id
from openpi_client.e0.artifacts import read_jsonl
from openpi_client.e0.artifacts import write_json
from openpi_client.e0.protocol import SUITES
from openpi_client.e0.protocol import summarize_attempts
from openpi_client.e0.protocol import validate_grid

from openpi.e0.config import read_manifest
from openpi.e0.jobs import ROOT
from openpi.e0.provenance import code_record


def _stop(process):
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def evaluate(
    actor_manifest,
    checkpoint,
    episode_manifest,
    output,
    devices=(0,),
    workers_per_device=4,
    platform="cyy_dev",
    port_base=9000,
    timeout=120,
    wall_timeout=86400,
):
    spec = read_manifest(actor_manifest)
    if spec["kind"] != "actor":
        raise ValueError("evaluation requires an actor manifest")
    rows = read_jsonl(episode_manifest)
    if not rows or any(r["training_seed"] != spec["seed"] for r in rows):
        raise ValueError("episode manifest belongs to a different training seed")
    if not spec["smoke"]:
        purposes = {r["purpose"] for r in rows}
        if len(purposes) != 1:
            raise ValueError("mixed episode purposes")
        purpose = next(iter(purposes))
        suites = SUITES if purpose == "development" and spec["stage"] == "sft" else ["libero_10"]
        groups = [0, 1, 2] if purpose == "final" else [0]
        if purpose not in ("development", "final", "rollout"):
            raise ValueError("non-smoke episode purpose required")
        validate_grid(rows, suites, groups, purpose)
    devices = list(map(int, devices))
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0 or workers_per_device < 1:
        raise ValueError("invalid GPU/worker partition")
    if platform not in ("cyy_dev", "a800"):
        raise ValueError(platform)
    checkpoint = Path(checkpoint)
    checkpoint_metadata = json.loads((checkpoint / "e0-checkpoint.json").read_text())
    if checkpoint_metadata["manifest_id"] != spec["manifest_id"]:
        raise ValueError("checkpoint and actor manifest differ")
    policy_id = content_id(checkpoint_metadata)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result_dir = output / "results"
    fingerprint = {"episode_manifest_id": content_id(rows), "policy_id": policy_id}
    binding = output / "evaluation-binding.json"
    if binding.exists() and json.loads(binding.read_text()) != fingerprint:
        raise ValueError("output belongs to a different checkpoint or evaluation manifest")
    write_json(binding, fingerprint)
    existing = [json.loads(p.read_text()) for p in sorted((result_dir / "attempts").glob("*.json"))]
    summary = summarize_attempts(rows, existing)
    if summary["complete"]:
        write_json(output / "summary.json", summary)
        return summary
    env = dict(os.environ, E0_PLATFORM=platform)
    processes = []
    handles = []
    servers = []
    workers = []
    session_id = uuid.uuid4().hex
    started = time.monotonic()
    record = {
        "session_id": session_id,
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "platform": platform,
        "devices": devices,
        "workers_per_device": workers_per_device,
        "code": code_record(ROOT),
        "policy_id": policy_id,
        "episode_manifest": str(episode_manifest),
        "versions": {n: importlib.metadata.version(n) for n in ("jax", "jaxlib", "numpy", "torch")},
        "hardware": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True
        ).splitlines(),
        "complete": False,
    }
    record_path = output / "execution-records" / (session_id + ".json")
    write_json(record_path, record)

    def spawn(args, child_env, label):
        handle = (output / f"{session_id}-{label}.log").open("w")
        handles.append(handle)
        process = subprocess.Popen(
            ["bash", str(ROOT / "scripts/e0.sh"), *map(str, args)],
            cwd=ROOT,
            env=child_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append(process)
        return process

    try:
        for i, device in enumerate(devices):
            servers.append(
                spawn(
                    ["serve", actor_manifest, "--checkpoint", checkpoint, "--port", port_base + i],
                    dict(env, CUDA_VISIBLE_DEVICES=str(device)),
                    f"server-{device}",
                )
            )
        from openpi_client.e0.rollout import Client
        from openpi_client.e0.rollout import InfrastructureError

        ready = set()
        deadline = time.monotonic() + 1800
        while len(ready) < len(devices):
            for i, server in enumerate(servers):
                if server.poll() is not None:
                    raise RuntimeError(f"policy server {i} exited")
                if i in ready:
                    continue
                try:
                    client = Client("127.0.0.1", port_base + i, timeout)
                    metadata = client.metadata
                    client.close()
                    if metadata["policy_id"] != policy_id:
                        raise ValueError("port is serving a different checkpoint")
                    write_json(output / f"{session_id}-server-{i}-metadata.json", metadata)
                    ready.add(i)
                except InfrastructureError:
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError("server warmup exceeded 30 minutes")
            if len(ready) < len(devices):
                time.sleep(2)
        shards = len(devices) * workers_per_device
        for i, device in enumerate(devices):
            for j in range(workers_per_device):
                shard = i * workers_per_device + j
                workers.append(
                    spawn(
                        [
                            "rollout",
                            episode_manifest,
                            "--output",
                            result_dir,
                            "--port",
                            port_base + i,
                            "--shard",
                            shard,
                            "--shards",
                            shards,
                            "--timeout",
                            timeout,
                        ],
                        dict(env, MUJOCO_EGL_DEVICE_ID=str(device)),
                        f"worker-{shard}",
                    )
                )
        while any(p.poll() is None for p in workers):
            if any(p.poll() is not None for p in servers):
                raise RuntimeError("policy server exited during evaluation")
            if any(p.poll() not in (None, 0) for p in workers):
                raise RuntimeError("simulation worker failed; inspect its log")
            if time.monotonic() - started > wall_timeout:
                raise TimeoutError("evaluation wall-time limit exceeded")
            time.sleep(2)
        if any(p.returncode for p in workers):
            raise RuntimeError("simulation worker failed")
        results = [json.loads(p.read_text()) for p in sorted((result_dir / "attempts").glob("*.json"))]
        summary = summarize_attempts(rows, results)
        write_json(output / "summary.json", summary)
        if not summary["complete"]:
            raise RuntimeError("evaluation is incomplete; no changed denominator")
        record["complete"] = True
        return summary
    except BaseException:
        record["error"] = traceback.format_exc()
        raise
    finally:
        for process in reversed(processes):
            _stop(process)
        for handle in handles:
            handle.close()
        record["elapsed_seconds"] = time.monotonic() - started
        record["allocated_gpu_seconds"] = record["elapsed_seconds"] * len(devices)
        write_json(record_path, record)
