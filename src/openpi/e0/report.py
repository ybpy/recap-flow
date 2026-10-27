"""Aggregate paired evaluation episodes without treating eval groups as training repeats."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from openpi_client.e0.artifacts import content_id
from openpi_client.e0.artifacts import read_jsonl
from openpi_client.e0.artifacts import write_json
from openpi_client.e0.protocol import summarize_attempts
from openpi_client.e0.protocol import training_seed_statistics
from openpi_client.e0.protocol import validate_grid


def final_report(index_path, output):
    index = json.loads(Path(index_path).read_text())
    methods = defaultdict(dict)
    task_rates = defaultdict(dict)
    raw = {}
    paired_manifests = {}
    observed_models = set()
    if not index["evaluations"]:
        raise ValueError("no final evaluations")
    for entry in index["evaluations"]:
        expected = read_jsonl(entry["episode_manifest"])
        validate_grid(expected, ["libero_10"], [0, 1, 2], "final")
        actual = [json.loads(p.read_text()) for p in sorted((Path(entry["results"]) / "attempts").glob("*.json"))]
        summary = summarize_attempts(expected, actual)
        if not summary["complete"]:
            raise ValueError(f"incomplete evaluation: {entry}")
        seed, method = entry["training_seed"], entry["method"]
        if any(row["training_seed"] != seed for row in expected):
            raise ValueError("wrong training-seed namespace")
        fingerprint = content_id(expected)
        if seed in paired_manifests and paired_manifests[seed] != fingerprint:
            raise ValueError("methods used different paired final manifests")
        paired_manifests[seed] = fingerprint
        selection = json.loads(Path(entry["selection"]).read_text())
        checkpoint = Path(selection["checkpoint"])
        checkpoint_record = json.loads((checkpoint / "e0-checkpoint.json").read_text())
        if (
            selection["kind"] != "actor"
            or selection["seed"] != seed
            or checkpoint_record["seed"] != seed
            or checkpoint_record["smoke"]
        ):
            raise ValueError("final report requires the selected formal actor from this training seed")
        identity = (seed, selection["stage"], selection["branch"])
        if identity in observed_models:
            raise ValueError("same trained model appears under multiple method names")
        observed_models.add(identity)
        policy_id = content_id(checkpoint_record)
        protocol = json.loads((Path(entry["results"]) / "protocol.json").read_text())
        if protocol["policy_id"] != policy_id:
            raise ValueError("final evaluation did not use the development-selected checkpoint")
        if seed not in (0, 1, 2):
            raise ValueError("final report accepts formal training seeds 0,1,2 only")
        if seed in methods[method]:
            raise ValueError("duplicate trained model evaluation")
        by_group = defaultdict(list)
        for row in actual:
            by_group[row["group"]].append(row)
        if set(by_group) != {0, 1, 2}:
            raise ValueError("final evaluation requires exactly 3 groups")
        group_scores = []
        for group in (0, 1, 2):
            specs = [r for r in expected if r["group"] == group]
            if len(specs) != 500:
                raise ValueError("each final group must contain 500 Long attempts")
            group_scores.append(summarize_attempts(specs, by_group[group])["macro_success"])
        methods[method][seed] = group_scores
        task_rates[method][seed] = summary["task_success"]
        raw[(method, seed)] = {r["episode_uid"]: float(r["success"]) for r in actual}
    expected_models = {
        (seed, stage, branch)
        for seed in (0, 1, 2)
        for stage, branch in (
            ("sft", "shared"),
            ("r1", "base_reset"),
            ("r1", "warm_start"),
            ("r2", "base_reset"),
            ("r2", "warm_start"),
        )
    }
    if observed_models != expected_models:
        raise ValueError("E0 final report requires all 15 selected models (3 seeds x SFT/R1/R2 branches)")
    results = {}
    for method, seeds in methods.items():
        if set(seeds) != {0, 1, 2}:
            raise ValueError(f"method missing training seeds: {method}")
        results[method] = training_seed_statistics(seeds)
        results[method]["per_task_by_training_seed"] = task_rates[method]
    paired = {}
    for left, right in index.get("comparisons", []):
        differences = {}
        for seed in (0, 1, 2):
            a, b = raw[(left, seed)], raw[(right, seed)]
            if a.keys() != b.keys():
                raise ValueError("methods used different evaluation episode manifests")
            differences[seed] = float(np.mean([b[uid] - a[uid] for uid in a]))
        paired[f"{right} - {left}"] = {
            "per_training_seed": differences,
            "mean": float(np.mean(list(differences.values()))),
            "sample_std": float(np.std(list(differences.values()), ddof=1)),
        }
    report = {
        "methods": results,
        "paired_differences": paired,
        "experimental_unit": "trained model seed; evaluation groups averaged within each model",
        "note": "official initial states reused; not an unseen-initial-state generalization claim",
    }
    write_json(output, report)
    return report
