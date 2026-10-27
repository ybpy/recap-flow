"""Selection requires every declared candidate and complete paired development evaluations."""

import json
from pathlib import Path

from openpi_client.e0.artifacts import content_id
from openpi_client.e0.artifacts import read_jsonl
from openpi_client.e0.artifacts import write_json
from openpi_client.e0.protocol import select_checkpoint
from openpi_client.e0.protocol import summarize_attempts
from openpi_client.e0.protocol import validate_grid

from openpi.e0.config import read_manifest


def actor(training_manifest, evaluations, output):
    spec = read_manifest(training_manifest)
    entries = json.loads(Path(evaluations).read_text())
    if sorted(e["updates"] for e in entries) != sorted(spec["checkpoint_updates"]):
        raise ValueError("must evaluate exactly the declared checkpoint candidates")
    expected_suites = (
        {"libero_spatial", "libero_object", "libero_goal", "libero_10"} if spec["stage"] == "sft" else {"libero_10"}
    )
    rows, paired = [], None
    for entry in entries:
        expected = read_jsonl(entry["episode_manifest"])
        if {r["suite"] for r in expected} != expected_suites or len(expected) != len(expected_suites) * 500:
            raise ValueError("development selection requires all declared tasks and all 50 initial states")
        validate_grid(expected, expected_suites, [0], "development")
        checkpoint = Path(spec["run_dir"]) / "candidates" / str(entry["updates"])
        identity = content_id(json.loads((checkpoint / "e0-checkpoint.json").read_text()))
        protocol = json.loads((Path(entry["results"]) / "protocol.json").read_text())
        if protocol["policy_id"] != identity:
            raise ValueError("candidate evaluated with a different policy")
        fingerprint = content_id(expected)
        if paired is not None and paired != fingerprint:
            raise ValueError("candidate checkpoints used different evaluation manifests")
        paired = fingerprint
        results = [json.loads(p.read_text()) for p in sorted((Path(entry["results"]) / "attempts").glob("*.json"))]
        report = summarize_attempts(expected, results)
        if not report["complete"]:
            raise ValueError("cannot select from incomplete evaluations")
        rows.append(dict(entry, macro_success=report["macro_success"]))
    winner = select_checkpoint(rows, "macro_success")
    checkpoint = Path(spec["run_dir"]) / "candidates" / str(winner["updates"])
    if not (checkpoint / "e0-checkpoint.json").exists():
        raise FileNotFoundError(checkpoint)
    write_json(
        output,
        {
            "kind": "actor",
            "seed": spec["seed"],
            "stage": spec["stage"],
            "branch": spec.get("branch"),
            "criterion": "macro_success; earliest checkpoint on exact tie",
            "candidates": rows,
            "checkpoint": str(checkpoint),
            "updates": winner["updates"],
            "manifest_id": spec["manifest_id"],
            "paired_episode_manifest_id": paired,
        },
    )


def critic(training_manifest, predictions, output):
    spec = read_manifest(training_manifest)
    rows, validation_id = [], None
    for path in predictions:
        report = json.loads((Path(path) / "predictions.json").read_text())
        checkpoint = Path(report["checkpoint"])
        metadata = json.loads((checkpoint / "e0-checkpoint.json").read_text())
        if metadata["manifest_id"] != spec["manifest_id"]:
            raise ValueError("wrong critic run")
        if validation_id is not None and validation_id != report["dataset_id"]:
            raise ValueError("critic candidates used different validation data")
        validation_id = report["dataset_id"]
        if len(report["task_ce"]) != 10:
            raise ValueError("critic CE must cover all ten Long tasks")
        rows.append({"updates": metadata["updates"], "macro_ce": report["macro_ce"], "checkpoint": str(checkpoint)})
    if sorted(r["updates"] for r in rows) != sorted(spec["checkpoint_updates"]):
        raise ValueError("must validate every declared critic candidate")
    winner = select_checkpoint(rows, "macro_ce", maximize=False)
    write_json(
        output,
        {
            "kind": "critic",
            "seed": spec["seed"],
            "stage": spec["stage"],
            "branch": spec.get("branch"),
            "criterion": "task-macro CE; earliest checkpoint on exact tie",
            "candidates": rows,
            "checkpoint": winner["checkpoint"],
            "updates": winner["updates"],
            "manifest_id": spec["manifest_id"],
            "validation_dataset_id": validation_id,
        },
    )
