"""Run the paired seed-42 B/C-clip actor experiment on one development node."""

import fcntl
import json
from pathlib import Path
import time

import numpy as np
from openpi_client.e0.artifacts import content_id, read_jsonl, write_json, write_jsonl
from openpi_client.e0.protocol import episodes, validate_grid

from openpi.e0.critic_study import P2, ROOT, SFT, cmd, read, stage_dir
from openpi.e0.jobs import training_config
from openpi.e0.reward_study import load_dataset

METHODS = {"B": "task_pool_max_clip", "C": "global_pool_max_clip"}


def seal(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f"sealed experiment input changed: {path}")
    write_json(path, value)


def prepare(work, study, status):
    if read(study / "status.json")["phase"] != "complete":
        raise ValueError("critic study must be complete")
    selection = read(P2 / "selection.json")
    if selection["updates"] != 40000:
        raise ValueError("the registered SFT reference must be the selected 40k EMA")
    bindings = {}
    for variant, method in METHODS.items():
        name = f"{method}-all_backbones"
        base = study / name
        critic_spec = read(base / "training.json")
        selected = read(base / "selection.json")
        pred = read(base / "train/predictions.json")
        if (
            selected["updates"] != 20000
            or pred["checkpoint"] != selected["checkpoint"]
            or critic_spec["freeze_mode"] != "all_backbones"
            or critic_spec["seed"] != 42
        ):
            raise ValueError("actor comparison requires the selected frozen 20k critics")
        bindings[variant] = dict(
            critic_manifest_id=critic_spec["manifest_id"],
            checkpoint=selected["checkpoint"],
            prediction_id=content_id(pred),
            threshold_id=read(study / "report" / name / "thresholds.json")["threshold_id"],
        )
    seal(
        work / "comparison.json",
        dict(
            critics=bindings,
            sft_selection=selection,
            seed=42,
            branch="base_reset",
            updates=60000,
            candidates=[30000, 45000, 60000],
            positive_percentile=70,
        ),
    )
    for phase, purpose, groups in (("development", "development", [0]), ("confirmation", "review", [0, 1, 2])):
        rows = list(
            episodes(
                f"actor-study/s42-bc-clip-v1/{phase}", ["libero_10"], groups=groups, purpose=purpose, training_seed=42
            )
        )
        path = work / (phase + "-episodes.jsonl")
        if path.exists() and read_jsonl(path) != rows:
            raise ValueError("paired episode manifest changed")
        write_jsonl(path, rows)
        validate_grid(rows, ["libero_10"], groups, purpose)
    manifests = {}
    for variant, method in METHODS.items():
        name = f"{method}-all_backbones"
        base = study / name
        source = Path(read(base / "training.json")["data_dir"])
        original = load_dataset(source)
        if (
            len(original["episodes"]) != 2193
            or sum(e["length"] for e in original["episodes"]) != 411854
            or sum(e["source"] == "demo" for e in original["episodes"]) != 1693
        ):
            raise ValueError("unexpected sealed R1 training pool")
        status("label-" + variant)
        output = work / ("data-" + variant)
        stage_dir(
            output,
            "e0-manifest.json",
            lambda tmp: cmd(
                "label",
                "--source",
                source,
                "--predictions",
                base / "train",
                "--output",
                tmp,
                "--critic-manifest",
                base / "training.json",
                "--thresholds",
                study / "report" / name / "thresholds.json",
            ),
        )
        request = dict(
            id=f"actor-study-s42-bc-clip-{variant}",
            kind="actor",
            stage="r1",
            branch="base_reset",
            seed=42,
            data_dir=str(output),
            num_train_steps=60000,
            checkpoint_updates=[30000, 45000, 60000],
        )
        path = work / variant / "request.json"
        seal(path, request)
        manifest = work / variant / "training.json"
        spec = training_config(path, manifest)
        bound = load_dataset(output)
        if (
            bound["parent_dataset_id"] != original["dataset_id"]
            or bound["critic_manifest_id"] != bindings[variant]["critic_manifest_id"]
            or bound["threshold_id"] != bindings[variant]["threshold_id"]
        ):
            raise ValueError("actor snapshot differs from the sealed comparison")
        smoke = dict(
            request,
            id=f"smoke-actor-bc-clip-{variant}",
            smoke=True,
            num_train_steps=20,
            checkpoint_interval=20,
            checkpoint_updates=[20],
        )
        seal(work / variant / "smoke-request.json", smoke)
        training_config(work / variant / "smoke-request.json", work / variant / "smoke.json")
        manifests[variant] = manifest
    a, b = [load_dataset(work / ("data-" + v)) for v in METHODS]
    fields = ("episode_uid", "episode_index", "length", "task_index", "source", "path")
    if [tuple(e[k] for k in fields) for e in a["episodes"]] != [tuple(e[k] for k in fields) for e in b["episodes"]]:
        raise ValueError("B/C sample order or membership differs")
    with (
        np.load(work / "data-B/advantage-diagnostics.npz") as a,
        np.load(work / "data-C/advantage-diagnostics.npz") as b,
    ):
        expert = a["expert"]
        np.testing.assert_array_equal(expert, b["expert"])
        np.testing.assert_array_equal(a["task_index"], b["task_index"])
        if not a["positive"][expert].all() or not b["positive"][expert].all():
            raise ValueError("expert frames must remain positive")
        write_json(
            work / "label-comparison.json",
            dict(
                frames=len(expert),
                expert_frames=int(expert.sum()),
                autonomous_frames=int((~expert).sum()),
                flipped_autonomous=int((a["positive"][~expert] != b["positive"][~expert]).sum()),
                parent_dataset_id=original["dataset_id"],
            ),
        )
    return manifests


def train(manifest):
    spec = read(manifest)
    directory = Path(spec["run_dir"])
    status = directory / "status.json"
    if status.exists() and read(status)["complete"]:
        if read(directory / "resolved-manifest.json") != spec:
            raise ValueError("completed actor manifest differs")
        return
    cmd("resume" if (directory / "resolved-manifest.json").exists() else "run", manifest)


def smoke(work, status):
    records = {}
    for variant in METHODS:
        status("smoke-" + variant)
        manifest = work / variant / "smoke.json"
        train(manifest)
        directory = Path(read(manifest)["run_dir"])
        records[variant] = dict(
            initial=read(directory / "initial-state-fingerprint.json"),
            final=read(directory / "state-fingerprint.json"),
            metrics=read_jsonl(directory / "metrics.jsonl"),
        )
    if records["B"]["initial"] != records["C"]["initial"]:
        raise ValueError("B/C initial parameters, optimizer, EMA or RNG differ")
    if records["B"]["final"] == records["C"]["final"]:
        raise ValueError("different conditions did not affect the smoke update")
    for a, b in zip(records["B"]["metrics"], records["C"]["metrics"], strict=True):
        if a["rng_key"] != b["rng_key"]:
            raise ValueError("paired training RNG differs")
        for key in ("task_frame_counts", "rollout_frames", "observation_action_sha256"):
            if a["exposure"][key] != b["exposure"][key]:
                raise ValueError(f"actor inputs differ beyond conditioning: {key}")
    write_json(
        work / "smoke.json",
        dict(
            passed=True,
            updates_per_variant=20,
            batch=256,
            fsdp_devices=4,
            initial_states_equal=True,
            noncondition_inputs_equal=True,
            conditions_affect_updates=True,
        ),
    )


def evaluate(manifest, checkpoint, episodes_path, output):
    cmd(
        "evaluate",
        "--actor-manifest",
        manifest,
        "--checkpoint",
        checkpoint,
        "--episode-manifest",
        episodes_path,
        "--output",
        output,
        "--platform",
        "pro5000",
        "--devices",
        *range(8),
        "--workers-per-device",
        4,
    )


def run(critic_study, output, prepare_only=False):
    study, work = Path(critic_study).resolve(), Path(output).resolve()
    if study == work or study in work.parents or work in study.parents:
        raise ValueError("actor comparison requires its own output directory")
    work.mkdir(parents=True, exist_ok=True)
    with (work / "study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execution = work / "execution.json"

        def check_execution():
            from openpi.e0.provenance import code_record, runtime_identity

            code = code_record(ROOT)
            seal(
                execution,
                dict(source_id=code["source_id"], commit=code["commit"], runtime_id=runtime_identity()["runtime_id"]),
            )

        def status(phase, **info):
            if execution.exists() and phase != "failed":
                check_execution()
            write_json(work / "status.json", dict(phase=phase, unix=time.time(), **info))

        try:
            manifests = prepare(work, study, status)
            if prepare_only:
                status("prepared")
                return
            check_execution()
            smoke(work, status)
            from openpi.e0.selection import actor

            for variant, manifest in manifests.items():
                status("train-" + variant, manifest=str(manifest))
                train(manifest)
                run_dir = Path(read(manifest)["run_dir"])
                entries = []
                for updates in (30000, 45000, 60000):
                    status(f"development-{variant}-{updates}")
                    out = work / variant / "development" / str(updates)
                    evaluate(manifest, run_dir / "candidates" / str(updates), work / "development-episodes.jsonl", out)
                    entries.append(
                        dict(
                            updates=updates,
                            episode_manifest=str(work / "development-episodes.jsonl"),
                            results=str(out / "results"),
                        )
                    )
                write_json(work / variant / "evaluations.json", entries)
                actor(manifest, work / variant / "evaluations.json", work / variant / "selection.json")
            selected = {v: (m, read(work / v / "selection.json")["checkpoint"]) for v, m in manifests.items()}
            selected["SFT"] = (SFT, read(work / "comparison.json")["sft_selection"]["checkpoint"])
            for variant, (manifest, checkpoint) in selected.items():
                status("confirmation-" + variant)
                evaluate(manifest, checkpoint, work / "confirmation-episodes.jsonl", work / variant / "confirmation")
            status("report")
            from openpi.e0.actor_report import report

            stage_dir(work / "report", "index.html", lambda tmp: report(work, tmp))
            status("complete", report=str(work / "report/index.html"))
        except BaseException as exc:
            prior = read(work / "status.json") if (work / "status.json").exists() else {}
            status("failed", stage=prior.get("phase"), error=repr(exc))
            raise
