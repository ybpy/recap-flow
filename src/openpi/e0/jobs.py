"""Build explicit E0 jobs. This module never submits cloud jobs."""

import json
from pathlib import Path
import shlex
import subprocess

from openpi.e0.platform import actor_fsdp
from openpi_client.e0.artifacts import content_id
from openpi_client.e0.artifacts import sha256_file
from openpi_client.e0.artifacts import write_json

RUNTIME = Path("/mnt/recap-flow-runtime")
ROOT = Path(__file__).resolve().parents[3]


def smoke_configs(output, platform="cyy_dev", name="e0-smoke"):
    if platform not in ("cyy_dev", "a800", "pro5000"):
        raise ValueError(platform)
    common = {
        "stage": "sft",
        "platform": platform,
        "seed": 42,
        "data_dir": str(RUNTIME / "datasets/sft-v1"),
        "dataset_manifest": str(RUNTIME / "datasets/sft-v1/e0-manifest.json"),
        "num_train_steps": 20,
        "warmup_steps": 2,
        "num_workers": 8 if platform == "pro5000" else 0,
        "checkpoint_interval": 10,
        "checkpoint_updates": [20],
        "smoke": True,
        "branch": "shared",
        "norm_assets": str(RUNTIME / "assets"),
    }
    actor = dict(
        common,
        id=f"{name}-{platform}-actor",
        kind="actor",
        batch_size=4 if platform == "cyy_dev" else 256,
        fsdp_devices=actor_fsdp(platform),
        num_train_steps=100 if platform == "pro5000" else 20,
        checkpoint_updates=[20, 100] if platform == "pro5000" else [20],
        initial_params="/mnt/models/pi05_base/params",
        initialization="pi05_base",
    )
    result = [actor]
    result.extend(
        dict(
            common,
            id=f"{name}-{platform}-critic-{mode}",
            kind="critic",
            batch_size=4 if platform == "cyy_dev" else 32,
            fsdp_devices=1,
            stage="r1",
            freeze_mode=mode,
            siglip_path=str(RUNTIME / "assets/critic/siglip-resumed.npz"),
            gemma_checkpoint_dir=str(RUNTIME / "assets/critic/gemma-pretrained"),
            tokenizer_path=str(RUNTIME / "assets/critic/tokenizer.model"),
        )
        for mode in ("all_backbones", "none")
    )
    paths = []
    for spec in result:
        spec["run_dir"] = str(RUNTIME / "runs" / spec["id"])
        path = Path(output) / (spec["id"] + ".json")
        if path.exists() and json.loads(path.read_text()) != spec:
            raise ValueError(f"refuse to replace different job {path}")
        write_json(path, spec)
        paths.append(str(path))
    return paths


def pilot_sft(output, platform="pro5000"):
    name = f"pilot-s42-sft-50k-{platform}"
    spec = {
        "id": name,
        "kind": "actor",
        "stage": "sft",
        "platform": platform,
        "seed": 42,
        "branch": "shared",
        "run_dir": str(RUNTIME / "runs" / name),
        "data_dir": str(RUNTIME / "datasets/sft-v1"),
        "dataset_manifest": str(RUNTIME / "datasets/sft-v1/e0-manifest.json"),
        "num_train_steps": 50000,
        "warmup_steps": 10000,
        "batch_size": 256,
        "fsdp_devices": actor_fsdp(platform),
        "num_workers": 8,
        "checkpoint_interval": 1000,
        "checkpoint_updates": [30000, 40000, 50000],
        "smoke": False,
        "norm_assets": str(RUNTIME / "assets"),
        "initial_params": "/mnt/models/pi05_base/params",
        "initialization": "pi05_base",
    }
    write_json(output, spec)
    return spec


def training_config(request, output):
    """Resolve a stage request only after its concrete data and selections exist."""
    from openpi.e0.config import read_manifest

    request = json.loads(Path(request).read_text())
    spec = dict(request)
    spec.setdefault("run_dir", str(RUNTIME / "runs" / spec["id"]))
    spec.setdefault("dataset_manifest", str(Path(spec["data_dir"]) / "e0-manifest.json"))
    spec.setdefault("norm_assets", str(RUNTIME / "assets"))
    spec.setdefault("smoke", False)
    spec.setdefault("batch_size", 256 if spec["kind"] == "actor" else 32)
    spec.setdefault("warmup_steps", 10000 if spec["kind"] == "actor" else 1000)
    spec.setdefault("platform", "pro5000")
    spec.setdefault("fsdp_devices", actor_fsdp(spec["platform"]) if spec["kind"] == "actor" else 1)
    spec.setdefault("num_workers", 8)
    spec.setdefault("checkpoint_interval", 1000)
    if spec["kind"] == "actor":
        if spec["stage"] == "sft" or spec["branch"] == "base_reset":
            spec.update(initialization="pi05_base", initial_params="/mnt/models/pi05_base/params")
        else:
            selection = json.loads(Path(spec["parent_selection"]).read_text())
            spec.update(initialization="ema", initial_params=str(Path(selection["checkpoint"]) / "params"))
    else:
        spec.setdefault("siglip_path", str(RUNTIME / "assets/critic/siglip-resumed.npz"))
        spec.setdefault("gemma_checkpoint_dir", str(RUNTIME / "assets/critic/gemma-pretrained"))
        spec.setdefault("tokenizer_path", str(RUNTIME / "assets/critic/tokenizer.model"))
    write_json(output, spec)
    resolved = read_manifest(output)
    write_json(output, resolved)
    return resolved


def freeze(calibration, output):
    """Require explicit completed pilot evidence; never invent calibrated budgets."""
    data = json.loads(Path(calibration).read_text())
    required = {
        "actor_updates",
        "actor_candidates",
        "critic_updates",
        "critic_candidates",
        "critic_freeze_mode",
        "evidence",
    }
    if not required <= data.keys():
        raise ValueError(f"missing calibration: {sorted(required - data.keys())}")
    if data["critic_freeze_mode"] not in ("all_backbones", "none"):
        raise ValueError("invalid critic freeze mode")
    if not data["evidence"]:
        raise ValueError("pilot evidence required")
    for path in data["evidence"]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    for stage in ("sft", "r1", "r2"):
        budget = data["actor_updates"][stage]
        candidates = data["actor_candidates"][stage]
        if (
            budget <= 10000
            or len(candidates) != 4
            or len(set(candidates)) != 4
            or any(x < 1 or x > budget for x in candidates)
        ):
            raise ValueError(f"invalid common actor budget/candidates for {stage}")
    for stage in ("r1", "r2"):
        budget = data["critic_updates"]
        window = data["critic_candidates"]
        budget = budget[stage] if isinstance(budget, dict) else budget
        window = window[stage] if isinstance(window, dict) else window
        if budget <= 1000 or not window or window != sorted(set(window)) or any(x < 1 or x > budget for x in window):
            raise ValueError(f"invalid critic budget/candidates for {stage}")
    data["evidence_sha256"] = {p: sha256_file(p) for p in data["evidence"]}
    data["base_protocol_sha256"] = sha256_file(ROOT / "e0/protocol.json")
    data["frozen_protocol_id"] = content_id(data)
    write_json(output, data)


def matrix(output):
    """Dependency graph, not runnable jobs before pilot calibration."""
    nodes = []
    for seed in (0, 1, 2):
        prefix = f"s{seed}"
        sft = f"{prefix}-sft"
        nodes.append({"id": sft, "kind": "actor", "stage": "sft", "seed": seed, "branch": "shared", "dependencies": []})
        r1 = f"{prefix}-r1-rollout"
        critic = f"{prefix}-r1-critic"
        labels = f"{prefix}-r1-labels"
        nodes.extend(
            [
                {"id": r1, "kind": "rollout", "seed": seed, "branch": "shared", "dependencies": [sft + "-selected"]},
                {
                    "id": critic,
                    "kind": "critic",
                    "stage": "r1",
                    "seed": seed,
                    "branch": "shared",
                    "dependencies": [r1, sft + "-development"],
                },
                {
                    "id": labels,
                    "kind": "labels",
                    "seed": seed,
                    "branch": "shared",
                    "dependencies": [critic + "-selected"],
                },
            ]
        )
        for branch in ("base_reset", "warm_start"):
            a1 = f"{prefix}-{branch}-r1"
            r2 = f"{prefix}-{branch}-r2-rollout"
            c2 = f"{prefix}-{branch}-r2-critic"
            l2 = f"{prefix}-{branch}-r2-labels"
            a2 = f"{prefix}-{branch}-r2"
            nodes.extend(
                [
                    {
                        "id": a1,
                        "kind": "actor",
                        "stage": "r1",
                        "seed": seed,
                        "branch": branch,
                        "dependencies": [labels, sft + "-selected"],
                    },
                    {"id": r2, "kind": "rollout", "seed": seed, "branch": branch, "dependencies": [a1 + "-selected"]},
                    {
                        "id": c2,
                        "kind": "critic",
                        "stage": "r2",
                        "seed": seed,
                        "branch": branch,
                        "dependencies": [r1, r2, sft + "-development", a1 + "-development"],
                    },
                    {"id": l2, "kind": "labels", "seed": seed, "branch": branch, "dependencies": [c2 + "-selected"]},
                    {
                        "id": a2,
                        "kind": "actor",
                        "stage": "r2",
                        "seed": seed,
                        "branch": branch,
                        "dependencies": [l2, a1 + "-selected"],
                    },
                ]
            )
    write_json(
        output,
        {
            "status": "dependency graph; instantiate jobs after calibration and dependencies",
            "nodes": nodes,
            "actor_runs": 15,
            "critic_runs": 9,
            "training_attempts": 4500,
            "development_attempts_if_four_candidates": 48000,
            "final_attempts": 22500,
        },
    )


def task_card(manifest, bindings, output):
    """Render a launchable card only when real image/mount bindings are supplied."""
    from openpi.e0.config import read_manifest

    spec = read_manifest(manifest)
    binding = json.loads(Path(bindings).read_text())
    image = binding["image"]
    if not image.get("uri") or not image.get("digest", "").startswith("sha256:"):
        raise ValueError("actual image URI and digest required")
    mounts = binding["mounts"]
    for mount in mounts:
        if not all(mount.get(k) for k in ("source", "container", "mode")):
            raise ValueError("each mount needs actual source, container path, and mode")
    operation = binding.get("operation", "train")
    resources = binding["resources"]
    platform = spec.get("platform", "a800")
    gpu_type = "PRO5000-72G" if platform == "pro5000" else "A800-80G"
    if operation in ("train", "validation") and (
        resources["gpu_count"] != 8 or resources["nodes"] != 1 or resources["gpu_type"] != gpu_type
    ):
        raise ValueError(f"E0 training/validation requires one node with 8 {gpu_type}")
    for required in (ROOT, Path("/mnt/models"), RUNTIME):
        if not any(required.is_relative_to(Path(mount["container"])) for mount in mounts):
            raise ValueError(f"task mounts do not cover {required}")
    if not any(Path(mount["container"]) == ROOT for mount in mounts):
        raise ValueError("mount the immutable code/environment release explicitly at /mnt/code/recap-flow")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("publish a clean committed code snapshot before generating a task card")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if binding["code_commit"] != commit:
        raise ValueError("binding code commit is not this release")
    if not binding.get("release_snapshot"):
        raise ValueError("immutable code/environment snapshot required")
    environment = {
        "E0_PLATFORM": platform,
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "RECAP_RUNTIME": str(RUNTIME),
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "OMP_NUM_THREADS": "4",
        "WANDB_MODE": "disabled",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
    }
    environment.update(binding.get("environment", {}))
    assignments = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in environment.items())
    base = f"cd {shlex.quote(str(ROOT))} && env {assignments} bash scripts/e0.sh"
    sealed = RUNTIME / "jobs/sealed" / (spec["id"] + "-" + spec["manifest_id"] + ".json")
    if sealed.exists() and json.loads(sealed.read_text()) != spec:
        raise ValueError("sealed manifest path already contains different parameters")
    write_json(sealed, spec)
    launch = base + " run " + shlex.quote(str(sealed))
    card = {
        "name": spec["id"],
        "stage": spec["stage"],
        "seed": spec["seed"],
        "branch": spec["branch"],
        "round": {"sft": 0, "r1": 1, "r2": 2}[spec["stage"]],
        "dependencies": binding.get("dependencies", []),
        "image": image,
        "code_commit": commit,
        "environment_id": content_id(
            {
                n: sha256_file(ROOT / "e0/environment" / n)
                for n in (
                    "train.lock",
                    "libero.lock",
                    "runtime-lock.json",
                    "critic-assets.lock.json",
                    "nccl-source.lock.json",
                    "nvidia-580.126.20.lock.json",
                )
            }
        ),
        "release_snapshot": binding["release_snapshot"],
        "resources": binding["resources"],
        "protocol": json.loads((ROOT / "e0/protocol.json").read_text()),
        "mounts": mounts,
        "environment": environment,
        "manifest": spec,
        "launch_command": launch,
        "outputs": {
            "run": spec["run_dir"],
            "metrics": spec["run_dir"] + "/metrics.jsonl",
            "candidates": spec["run_dir"] + "/candidates",
            "recovery": spec["run_dir"] + "/recovery",
        },
        "acceptance": [
            "status.complete=true",
            "finite training metrics",
            "declared EMA candidates and assets present",
            "run provenance matches release, dataset and environment",
        ],
        "resume_command": base + " resume " + shlex.quote(str(sealed)),
    }
    schema = json.loads((ROOT / "e0/task-card.schema.json").read_text())
    if set(schema["required"]) - card.keys():
        raise ValueError("incomplete task card")
    if set(schema["properties"]["resources"]["required"]) - card["resources"].keys():
        raise ValueError("incomplete resource specification")
    operation = binding.get("operation", "train")
    if operation == "evaluate":
        evaluation = binding["evaluation"]
        args = [
            "evaluate",
            "--actor-manifest",
            str(sealed),
            "--checkpoint",
            evaluation["checkpoint"],
            "--episode-manifest",
            evaluation["episode_manifest"],
            "--output",
            evaluation["output"],
            "--platform",
            platform,
            "--devices",
            *map(str, evaluation["devices"]),
            "--workers-per-device",
            str(evaluation["workers_per_device"]),
        ]
        card["launch_command"] = base + " " + shlex.join(args)
        card["resume_command"] = card["launch_command"]
        card["manifest"] = {"actor": spec, "evaluation": evaluation}
        card["outputs"] = {"root": evaluation["output"], "summary": evaluation["output"] + "/summary.json"}
        card["acceptance"] = [
            "summary.complete=true",
            "full declared episode denominator",
            "all results bound to the requested checkpoint",
        ]
    elif operation == "validation":
        validation_output = binding["validation_output"]
        card["launch_command"] = (
            base + " smoke --platform " + platform + " --output " + shlex.quote(validation_output) + " --phase all"
        )
        card["resume_command"] = card["launch_command"]
        card["manifest"] = {
            "validation_profile": platform,
            "actual_updates": {"actor": 100 if platform == "pro5000" else 20, "critic": 20},
            "actor_batch": 256,
            "critic_batch": 32,
            "actor_fsdp_devices": actor_fsdp(platform),
            "critic_fsdp_devices": 1,
            "critic_modes": ["all_backbones", "none"],
            "recovery": "10 + 10 versus continuous 20",
            "attempts_per_task": 2,
        }
        card["outputs"] = {"root": validation_output}
        card["acceptance"] = [
            "preflight-passed.json",
            "training-passed.json",
            "recovery-passed.json",
            "closed-loop-passed.json",
        ]
    elif operation != "train":
        raise ValueError("unsupported task-card operation")
    write_json(output, card)
    return card
