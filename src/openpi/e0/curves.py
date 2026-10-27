"""Build learning curves from selected evaluations and explicitly listed cumulative costs."""

import json
from pathlib import Path

import numpy as np
from openpi_client.e0.artifacts import read_jsonl
from openpi_client.e0.artifacts import write_json


def _training_cost(run):
    run = Path(run)
    spec = json.loads((run / "resolved-manifest.json").read_text())
    status = json.loads((run / "status.json").read_text())
    if not status["complete"]:
        raise ValueError(f"incomplete training run: {run}")
    rows = read_jsonl(run / "metrics.jsonl")
    if [r["updates"] for r in rows] != list(range(1, status["completed_updates"] + 1)):
        raise ValueError("missing/duplicated training updates in cost ledger")
    if any("session_id" not in row for row in rows):
        raise ValueError("legacy smoke metrics lack session IDs and cannot be used for formal cost accounting")
    sessions = {}
    for row in rows:
        sessions[row["session_id"]] = row["elapsed_seconds"]
    elapsed = sum(sessions.values())
    return {
        "training_loop_gpu_seconds": elapsed * rows[-1]["exposure"]["gpu_count"],
        "sampled_frames": sum(r["exposure"]["global_batch"] for r in rows),
        "actor_updates": len(rows) if spec["kind"] == "actor" else 0,
        "critic_updates": len(rows) if spec["kind"] == "critic" else 0,
    }


def aggregate_points(points, *, formal=True):
    """The independent unit is a trained seed, not an evaluation group."""
    from collections import defaultdict

    groups = defaultdict(dict)
    for point in points:
        key = (point["method"], point["stage"])
        seed = point["training_seed"]
        if seed in groups[key]:
            raise ValueError("duplicate training seed at a curve point")
        if point["stage"] not in ("sft", "r1", "r2"):
            raise ValueError("unknown curve stage")
        groups[key][seed] = point
    result = []
    for (method, stage), seeds in sorted(groups.items()):
        if formal and set(seeds) != {0, 1, 2}:
            raise ValueError("formal curve points require training seeds 0,1,2")
        record = {"method": method, "stage": stage, "training_repeats": len(seeds)}
        for metric in (
            "success",
            "cumulative_environment_actions",
            "training_loop_gpu_hours",
            "measured_execution_gpu_hours",
        ):
            values = np.asarray([point[metric] for point in seeds.values()], float)
            if not np.isfinite(values).all():
                raise ValueError("nonfinite curve metric")
            record[metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
            }
        result.append(record)
    return result


def curves(index_path, output):
    index = json.loads(Path(index_path).read_text())
    points = []
    for entry in index["points"]:
        runs = entry["cumulative_training_runs"]
        collections = entry["cumulative_training_collections"]
        if len(set(runs)) != len(runs) or len(set(collections)) != len(collections):
            raise ValueError("shared SFT/R1 costs must be counted once per lineage")
        cost = [_training_cost(run) for run in runs]
        summaries = [json.loads((Path(p) / "summary.json").read_text()) for p in collections]
        if not all(s["complete"] for s in summaries):
            raise ValueError("incomplete collection cost record")
        evaluation = json.loads(Path(entry["evaluation_summary"]).read_text())
        if not evaluation["complete"]:
            raise ValueError("incomplete evaluation")
        measured = []
        for directory in entry.get("measured_execution_directories", []):
            measured.extend(json.loads(p.read_text()) for p in (Path(directory) / "execution-records").glob("*.json"))
        points.append(
            dict(
                entry,
                success=evaluation["macro_success"],
                cumulative_environment_actions=sum(s["policy_steps"] for s in summaries),
                training_loop_gpu_hours=sum(c["training_loop_gpu_seconds"] for c in cost) / 3600,
                sampled_frames=sum(c["sampled_frames"] for c in cost),
                measured_execution_gpu_hours=sum(r["allocated_gpu_seconds"] for r in measured) / 3600,
            )
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate_points(points, formal=index.get("formal", True))
    write_json(
        output / "curves.json",
        {
            "points": points,
            "aggregates": aggregates,
            "cost_note": "Training-loop time includes compilation from the first batch; excludes process startup/model loading. Measured evaluation/rollout allocation time is separate; no estimated FLOPs.",
        },
    )
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    for x, label in (
        ("cumulative_environment_actions", "Cumulative training environment actions"),
        ("training_loop_gpu_hours", "Cumulative training-loop GPU hours"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        for method in sorted({p["method"] for p in points}):
            selected = sorted(
                [p for p in aggregates if p["method"] == method], key=lambda p: {"sft": 0, "r1": 1, "r2": 2}[p["stage"]]
            )
            ax.errorbar(
                [p[x]["mean"] for p in selected],
                [p["success"]["mean"] for p in selected],
                xerr=[p[x]["sample_std"] or 0 for p in selected],
                yerr=[p["success"]["sample_std"] or 0 for p in selected],
                fmt="-o",
                capsize=3,
                label=method,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Task-macro success rate")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / (x + ".png"), dpi=180)
        plt.close(fig)
    return points
