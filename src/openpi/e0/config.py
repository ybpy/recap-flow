"""Resolve explicit manifests into pistar configurations."""

import dataclasses
import json
from pathlib import Path

from openpi_client.e0.artifacts import content_id


def read_manifest(path):
    spec = json.loads(Path(path).read_text())
    required = (
        "id",
        "kind",
        "stage",
        "seed",
        "run_dir",
        "data_dir",
        "dataset_manifest",
        "batch_size",
        "num_train_steps",
        "warmup_steps",
        "fsdp_devices",
        "num_workers",
        "checkpoint_interval",
        "checkpoint_updates",
        "smoke",
    )
    missing = set(required) - set(spec)
    if missing:
        raise ValueError(f"missing manifest fields: {sorted(missing)}")
    if spec["kind"] not in ("actor", "critic"):
        raise ValueError("training kind must be actor or critic")
    if spec["stage"] not in ("sft", "r1", "r2"):
        raise ValueError("stage must be sft/r1/r2")
    branch = spec.get("branch")
    if spec["stage"] == "sft":
        if spec["kind"] != "actor" or branch != "shared":
            raise ValueError("SFT must be a shared actor")
    elif spec["kind"] == "critic" and spec["stage"] == "r1":
        if branch != "shared":
            raise ValueError("R1 critic must be shared")
    elif branch not in ("base_reset", "warm_start"):
        raise ValueError("RECAP actor/R2 critic requires an explicit branch")
    for key in ("batch_size", "num_train_steps", "warmup_steps", "fsdp_devices", "checkpoint_interval"):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(f"invalid {key}")
    if spec["num_workers"] < 0:
        raise ValueError("invalid worker count")
    if not spec["checkpoint_updates"] or spec["checkpoint_updates"] != sorted(set(spec["checkpoint_updates"])):
        raise ValueError("candidate updates must be nonempty, sorted and unique")
    if not spec["smoke"]:
        expected_batch = 256 if spec["kind"] == "actor" else 32
        expected_warmup = 10000 if spec["kind"] == "actor" else 1000
        if spec["batch_size"] != expected_batch or spec["warmup_steps"] != expected_warmup:
            raise ValueError("non-smoke batch/warmup violates E0 protocol")
        if spec["seed"] != 42:
            if spec["seed"] not in (0, 1, 2):
                raise ValueError("formal training seeds must be 0,1,2")
            frozen = json.loads(Path(spec["frozen_protocol"]).read_text())
            claimed = frozen.pop("frozen_protocol_id")
            if claimed != content_id(frozen):
                raise ValueError("frozen protocol checksum mismatch")
            spec["frozen_protocol_id"] = claimed
            if spec["kind"] == "actor":
                steps = frozen["actor_updates"][spec["stage"]]
                candidates = frozen["actor_candidates"][spec["stage"]]
            else:
                budget = frozen["critic_updates"]
                window = frozen["critic_candidates"]
                steps = budget[spec["stage"]] if isinstance(budget, dict) else budget
                candidates = window[spec["stage"]] if isinstance(window, dict) else window
                if spec["freeze_mode"] != frozen["critic_freeze_mode"]:
                    raise ValueError("critic freeze mode differs from frozen protocol")
            if spec["num_train_steps"] != steps or spec["checkpoint_updates"] != candidates:
                raise ValueError("training budget/candidates differ from frozen protocol")
    if spec["num_train_steps"] <= spec["warmup_steps"]:
        raise ValueError("training budget must exceed warmup")
    if any(s < 1 or s > spec["num_train_steps"] for s in spec["checkpoint_updates"]):
        raise ValueError("invalid checkpoint candidates")
    for key, expected in (("H", 10), ("K", 5), ("flow_steps", 10), ("adv_guidance_beta", 1.0)):
        if key in spec and spec[key] != expected:
            raise ValueError(f"{key} differs from E0 protocol")
    from openpi.e0.platform import actor_fsdp

    platform = spec.get("platform", "a800")
    expected_fsdp = actor_fsdp(platform) if spec["kind"] == "actor" else 1
    if not spec["smoke"] and spec["fsdp_devices"] != expected_fsdp:
        raise ValueError(f"{platform} {spec['kind']} fsdp_devices must be {expected_fsdp}")
    data_path = Path(spec["dataset_manifest"])
    if data_path.resolve() != (Path(spec["data_dir"]) / "e0-manifest.json").resolve():
        raise ValueError("dataset manifest must belong to the actual training data directory")
    if not data_path.is_file():
        raise FileNotFoundError(f"audited dataset manifest not ready: {data_path}")
    dataset = json.loads(data_path.read_text())
    if dataset["dataset_id"] != content_id({k: v for k, v in dataset.items() if k != "dataset_id"}):
        raise ValueError("dataset manifest content ID mismatch")
    if spec.get("dataset_id", dataset["dataset_id"]) != dataset["dataset_id"]:
        raise ValueError("dataset version differs from the sealed task manifest")
    spec["dataset_id"] = dataset["dataset_id"]
    if spec["kind"] == "actor":
        for name in ("initialization", "initial_params", "norm_assets"):
            if name not in spec:
                raise ValueError(f"actor requires explicit {name}")
        expected_init = "pi05_base" if spec["stage"] == "sft" or branch == "base_reset" else "ema"
        if spec["initialization"] != expected_init:
            raise ValueError("actor initialization contradicts stage/branch")
        if spec["stage"] != "sft" and not dataset.get("critic_checkpoint"):
            raise ValueError("RECAP actor requires a freshly critic-labeled data snapshot")
        if spec["initialization"] == "pi05_base":
            if Path(spec["initial_params"]).resolve() != Path("/mnt/models/pi05_base/params").resolve():
                raise ValueError("base reset must use the fixed pi05_base")
        elif spec["initialization"] == "ema":
            provenance = json.loads((Path(spec["initial_params"]).parent / "e0-checkpoint.json").read_text())
            if provenance["parameter_kind"] != "EMA" or provenance["seed"] != spec["seed"]:
                raise ValueError("warm start requires this training seed's selected EMA")
            expected_previous = "sft" if spec["stage"] == "r1" else "r1"
            if provenance["stage"] != expected_previous:
                raise ValueError("warm-start checkpoint is from the wrong round")
            if spec["stage"] == "r2" and provenance["branch"] != spec["branch"]:
                raise ValueError("R2 warm start must inherit its own R1 branch")
            selected = json.loads(Path(spec["parent_selection"]).read_text())
            if Path(selected["checkpoint"]).resolve() != Path(spec["initial_params"]).parent.resolve():
                raise ValueError("warm start must use the selected previous-stage checkpoint")
        else:
            raise ValueError("unknown initialization")
        from openpi_client.e0.artifacts import sha256_file

        actual_norm = sha256_file(Path(spec["norm_assets"]) / "libero/norm_stats.json")
        if spec.get("norm_stats_sha256", actual_norm) != actual_norm:
            raise ValueError("normalization differs from the sealed task manifest")
        spec["norm_stats_sha256"] = actual_norm
        frozen_norm = Path("/mnt/recap-flow-runtime/assets/libero/norm_stats.json")
        if spec["norm_stats_sha256"] != sha256_file(frozen_norm):
            raise ValueError("LIBERO norm stats differ from the audited frozen statistics")
    claimed = spec.pop("manifest_id", None)
    spec["manifest_id"] = content_id(spec)
    if claimed is not None and claimed != spec["manifest_id"]:
        raise ValueError("sealed task manifest checksum mismatch")
    return spec


def actor_config(spec, *, inference=False):
    from openpi.training import config as c
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    baseline = c.get_config("pi05_star_libero")
    return dataclasses.replace(
        baseline,
        name="e0_pi05",
        project_name="recap-flow",
        exp_name=spec["id"],
        model=dataclasses.replace(baseline.model, action_horizon=10, adv_guidance_beta=1.0),
        data=c.LeRobotLiberoDataConfig(
            repo_id=spec["data_dir"],
            assets=c.AssetsConfig(assets_dir=spec["norm_assets"], asset_id="libero"),
            base_config=c.DataConfig(prompt_from_task=True, local_data_dir=spec["data_dir"]),
            extra_delta_transform=False,
            adv_ind_dropout=not inference,
        ),
        seed=spec["seed"],
        batch_size=spec["batch_size"],
        num_workers=spec["num_workers"],
        num_train_steps=spec["num_train_steps"],
        fsdp_devices=spec["fsdp_devices"],
        weight_loader=weight_loaders.CheckpointWeightLoader(spec["initial_params"]),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=spec["warmup_steps"],
            peak_lr=5e-5,
            decay_lr=5e-5,
            decay_steps=max(spec["num_train_steps"], spec["warmup_steps"] + 1),
        ),
        ema_decay=0.999,
        wandb_enabled=False,
        pytorch_weight_path=None,
        policy_metadata={"protocol": "e0-v1", "H": 10, "K": 5, "flow_steps": 10, "adv_guidance_beta": 1.0},
    )


def validate_lineage(spec, dataset):
    sources = dataset["episodes"]
    baseline = json.loads(Path("/mnt/recap-flow-runtime/datasets/sft-v1/e0-manifest.json").read_text())
    expected = {(e["episode_uid"], e["length"], e["sha256"]) for e in baseline["episodes"]}
    actual = {(e["episode_uid"], e["length"], e["sha256"]) for e in sources if e["source"] == "demo"}
    if actual != expected or sum(e["source"] == "demo" for e in sources) != len(expected):
        raise ValueError("E0 must retain every audited demo, unchanged, in every training pool")
    if not spec["smoke"] and any(e.get("smoke", False) for e in sources):
        raise ValueError("smoke trajectories cannot enter pilot or formal training")
    if any(e["source"] not in ("demo", "rollout", "intervention") for e in sources):
        raise ValueError("development/final/research or unknown data leaked into training")
    if spec["stage"] == "sft" and any(e["source"] != "demo" for e in sources):
        raise ValueError("SFT must use demos only")
    for e in sources:
        if e["source"] == "rollout":
            if e.get("training_seed") != spec["seed"]:
                raise ValueError("rollout from another training seed")
            round_index = e["round"]
            if round_index not in (1, 2):
                raise ValueError("only R1/R2 rollout data are eligible")
            if spec["stage"] == "r1" and round_index != 1:
                raise ValueError("future-round rollout leakage")
            if round_index == 1 and e["branch"] != "shared":
                raise ValueError("R1 pool must be shared")
            if round_index == 2 and e["branch"] != spec["branch"]:
                raise ValueError("R2 must use its own branch")
        if e["source"] == "rollout" and e.get("branch") not in ("shared", spec.get("branch")):
            raise ValueError(f"cross-branch rollout leakage: {e['episode_uid']}")
