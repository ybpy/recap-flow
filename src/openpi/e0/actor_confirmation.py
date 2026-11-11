"""Continue sealed actor confirmations after training and selection have completed."""

import fcntl
from pathlib import Path
import time

from openpi_client.e0.artifacts import read_jsonl, sha256_file, write_json
from openpi_client.e0.protocol import validate_grid

from openpi.e0.actor_study import evaluate, seal
from openpi.e0.config import read_manifest
from openpi.e0.critic_study import ROOT, SFT, read, stage_dir
from openpi.e0.provenance import code_record, runtime_identity
from openpi.e0.selection import actor


def selected_inputs(work):
    """Recheck completed training and reproduce selection without changing it."""
    selected = {}
    files = [work / "execution.json", work / "comparison.json", work / "confirmation-episodes.jsonl"]
    validate_grid(read_jsonl(files[-1]), ["libero_10"], [0, 1, 2], "review")
    for variant in ("B", "C"):
        manifest = work / variant / "training.json"
        spec = read_manifest(manifest)
        run_dir = Path(spec["run_dir"])
        state = read(run_dir / "status.json")
        if (
            not state["complete"]
            or state["completed_updates"] != 60000
            or spec["num_train_steps"] != 60000
            or spec["seed"] != 42
            or spec["stage"] != "r1"
            or spec["branch"] != "base_reset"
            or read(run_dir / "resolved-manifest.json") != spec
        ):
            raise ValueError("confirmation requires both sealed 60k actor runs to be complete")
        selection_path = work / variant / "selection.json"
        selection = read(selection_path)
        audit = work / variant / "confirmation-selection-audit.json"
        actor(manifest, work / variant / "evaluations.json", audit)
        if read(audit) != selection:
            raise ValueError("confirmation selection differs from complete paired development results")
        selected[variant] = (manifest, selection["checkpoint"])
        files.extend([manifest, selection_path, audit, run_dir / "status.json", run_dir / "resolved-manifest.json"])
    sft = read(work / "comparison.json")["sft_selection"]
    if sft["updates"] != 40000:
        raise ValueError("confirmation requires the sealed SFT 40k reference")
    selected["SFT"] = (SFT, sft["checkpoint"])
    files.append(SFT)
    for manifest, checkpoint in selected.values():
        spec = read_manifest(manifest)
        path = Path(checkpoint) / "e0-checkpoint.json"
        metadata = read(path)
        if metadata["manifest_id"] != spec["manifest_id"] or metadata["parameter_kind"] != "EMA":
            raise ValueError("confirmation checkpoint does not match its training manifest")
        files.append(path)
    return selected, files


def run(output):
    work = Path(output).resolve()
    with (work / "study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            selected, files = selected_inputs(work)
            training_execution = read(work / "execution.json")

            def check_binding():
                runtime = runtime_identity()["runtime_id"]
                if runtime != training_execution["runtime_id"]:
                    raise ValueError("confirmation runtime differs from the frozen training environment")
                code = code_record(ROOT)
                seal(work / "confirmation-execution.json", {
                    "training_execution": training_execution,
                    "commit": code["commit"],
                    "source_id": code["source_id"],
                    "runtime_id": runtime,
                    "inputs": {str(p): sha256_file(p) for p in files},
                })

            def status(phase, **info):
                check_binding()
                write_json(work / "status.json", dict(phase=phase, unix=time.time(), **info))

            for variant, (manifest, checkpoint) in selected.items():
                status("confirmation-" + variant)
                evaluate(manifest, checkpoint, work / "confirmation-episodes.jsonl", work / variant / "confirmation")
            status("report")
            from openpi.e0.actor_report import report

            stage_dir(work / "report", "index.html", lambda tmp: report(work, tmp))
            status("complete", report=str(work / "report/index.html"))
        except BaseException as exc:
            prior = read(work / "status.json")
            write_json(work / "status.json", dict(phase="failed", stage=prior.get("phase"), error=repr(exc), unix=time.time()))
            raise
