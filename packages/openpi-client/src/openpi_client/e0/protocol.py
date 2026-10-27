"""Frozen E0 definitions. No simulator or training-framework imports."""

from __future__ import annotations
import hashlib
import json
from collections import defaultdict
import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LIMITS = dict(zip(SUITES, (220, 280, 300, 520)))
H, K, FLOW_STEPS, WAIT_STEPS, LOOKAHEAD = 10, 5, 10, 10, 50


def stable_seed(*parts) -> int:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def episode_id(namespace, suite, task, initial_state, group=0):
    return f"{namespace}/{suite}/task-{task:02d}/init-{initial_state:02d}/group-{group}"


def episodes(namespace, suites=SUITES, groups=(0,), attempts=50, purpose="smoke", training_seed=42):
    if not 1 <= attempts <= 50:
        raise ValueError("attempts must be 1..50; formal runs require 50")
    for suite in suites:
        for task in range(10):
            for initial_state in range(attempts):
                for group in groups:
                    uid = episode_id(namespace, suite, task, initial_state, group)
                    yield {
                        "episode_uid": uid,
                        "namespace": namespace,
                        "suite": suite,
                        "purpose": purpose,
                        "training_seed": training_seed,
                        "task_id": task,
                        "initial_state_id": initial_state,
                        "group": group,
                        "env_seed": stable_seed(uid, "environment"),
                        "sampling_seed": stable_seed(uid, "action"),
                        "max_policy_steps": LIMITS[suite],
                    }


def trajectory_labels(length: int, success: bool):
    if length < 1:
        raise ValueError("zero-action attempts belong in the attempt ledger, not the training dataset")
    reward = np.zeros(length, np.float32)
    reward[-1] = float(success)
    reward_label = np.full(length, -1.0 / length, np.float32)
    reward_label[-1] = 0.0 if success else -1.0
    value_label = -(length - 1 - np.arange(length, dtype=np.float32)) / length
    if not success:
        value_label.fill(-1)
    return {"reward": reward, "reward_label": reward_label, "value_label": value_label}


def advantages(rewards, values, lookahead=LOOKAHEAD):
    r, v = np.asarray(rewards, np.float64), np.asarray(values, np.float64)
    if r.ndim != 1 or r.shape != v.shape or not len(r) or lookahead < 1:
        raise ValueError("expected nonempty aligned one-dimensional reward/value arrays")
    if not np.isfinite(r).all() or not np.isfinite(v).all():
        raise ValueError("nonfinite reward or critic prediction")
    cumulative = np.concatenate(([0.0], np.cumsum(r)))
    end = np.minimum(np.arange(len(r)) + lookahead, len(r))
    bootstrap = np.concatenate((v, [0.0]))[end]
    return cumulative[end] - cumulative[:-1] + bootstrap - v


def condition_labels(task_ids, advantage, expert):
    tasks, a, experts = np.asarray(task_ids), np.asarray(advantage, np.float64), np.asarray(expert, bool)
    if not (tasks.shape == a.shape == experts.shape):
        raise ValueError("misaligned label inputs")
    if not np.isfinite(a[~experts]).all():
        raise ValueError("nonfinite autonomous advantage")
    positive = np.ones(len(a), bool)
    diagnostics = {}
    for task in np.unique(tasks[~experts]):
        mask = (tasks == task) & ~experts
        threshold = float(np.percentile(a[mask], 70, method="linear"))
        positive[mask] = a[mask] >= threshold
        diagnostics[str(task)] = {
            "threshold": threshold,
            "frames": int(mask.sum()),
            "positive_fraction": float(positive[mask].mean()),
        }
    return np.where(positive, "positive", "negative"), diagnostics


def summarize_attempts(expected, results):
    expected = list(expected)
    if not expected:
        raise ValueError("empty evaluation manifest")
    if len({r["episode_uid"] for r in expected}) != len(expected):
        raise ValueError("duplicate expected episode IDs")
    expected = {row["episode_uid"]: row for row in expected}
    seen = {}
    for row in results:
        uid = row["episode_uid"]
        if uid not in expected or uid in seen:
            raise ValueError(f"unexpected or duplicate episode: {uid}")
        if row["status"] not in ("ok", "policy_failure", "infrastructure_error"):
            raise ValueError(f"unclassified attempt status: {row['status']}")
        for key in (
            "suite",
            "task_id",
            "initial_state_id",
            "group",
            "env_seed",
            "sampling_seed",
            "max_policy_steps",
            "purpose",
            "training_seed",
            "namespace",
        ):
            if key in expected[uid] and row.get(key) != expected[uid][key]:
                raise ValueError(f"episode configuration changed: {uid}/{key}")
        if row["status"] != "ok" and row["success"]:
            raise ValueError("a failed attempt cannot be successful")
        seen[uid] = row
    missing = sorted(set(expected) - set(seen))
    incomplete = sorted(uid for uid, row in seen.items() if row["status"] == "infrastructure_error")
    if missing or incomplete:
        return {"complete": False, "expected": len(expected), "missing": missing, "incomplete": incomplete}
    tasks = defaultdict(list)
    for uid, row in seen.items():
        spec = expected[uid]
        tasks[f"{spec['suite']}/{spec['task_id']}"].append(float(row["success"]))
    rates = {key: float(np.mean(value)) for key, value in sorted(tasks.items())}
    return {
        "complete": True,
        "attempts": len(seen),
        "task_success": rates,
        "macro_success": float(
            sum(__import__("fractions").Fraction(int(sum(value)), len(value)) for value in tasks.values()) / len(tasks)
        ),
        "policy_steps": sum(row["policy_steps"] for row in seen.values()),
    }


def select_checkpoint(rows, metric, maximize=True):
    if not rows:
        raise ValueError("no eligible checkpoints")
    return min(rows, key=lambda row: ((-1 if maximize else 1) * row[metric], row["updates"]))


def training_seed_statistics(rates):
    # Each mapping value contains the evaluation-group scores for ONE trained model.
    per_seed = {str(seed): float(np.mean(groups)) for seed, groups in rates.items()}
    values = list(per_seed.values())
    return {
        "per_training_seed": per_seed,
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "training_repeats": len(values),
    }


def validate_grid(rows, suites, groups, purpose):
    rows = list(rows)
    expected = {
        (suite, task, state, group) for suite in suites for task in range(10) for state in range(50) for group in groups
    }
    actual = {(r["suite"], r["task_id"], r["initial_state_id"], r["group"]) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("evaluation must cover the exact task/initial-state/group grid")
    for row in rows:
        if row.get("purpose") != purpose:
            raise ValueError("evaluation namespace purpose mismatch")
        if row["max_policy_steps"] != LIMITS[row["suite"]]:
            raise ValueError("evaluation horizon changed")
        if row["env_seed"] != stable_seed(row["episode_uid"], "environment") or row["sampling_seed"] != stable_seed(
            row["episode_uid"], "action"
        ):
            raise ValueError("evaluation seed derivation changed")
