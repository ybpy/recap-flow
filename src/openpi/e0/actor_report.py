"""Paired actor success rates with initial-state cluster bootstrap."""

import csv
import html
import json
from pathlib import Path

import numpy as np
from openpi_client.e0.artifacts import content_id, read_jsonl, write_json
from openpi_client.e0.protocol import summarize_attempts, validate_grid

from openpi.e0.critic_gallery import video

PAIRS = (("B", "C"), ("B", "SFT"), ("C", "SFT"))


def interval(values, draws):
    """Resample initial states within task, retaining all three evaluation seeds."""
    states = np.asarray(values, np.float64).mean(axis=2)
    boot = np.take_along_axis(states[:, None, :], draws, axis=2).mean(axis=2)
    task = {
        str(t): dict(mean=float(states[t].mean()), ci95=np.percentile(boot[t], [2.5, 97.5]).tolist()) for t in range(10)
    }
    return dict(mean=float(states.mean()), ci95=np.percentile(boot.mean(axis=0), [2.5, 97.5]).tolist(), by_task=task)


def report(work, dest):
    work, dest = Path(work), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    expected = read_jsonl(work / "confirmation-episodes.jsonl")
    validate_grid(expected, ["libero_10"], [0, 1, 2], "review")
    ledgers, matrices, selections, models = {}, {}, {}, {}
    for name in ("B", "C", "SFT"):
        selection = (
            json.loads((work / "comparison.json").read_text())["sft_selection"]
            if name == "SFT"
            else json.loads((work / name / "selection.json").read_text())
        )
        selections[name] = selection
        root = work / name / "confirmation/results"
        policy = content_id(json.loads((Path(selection["checkpoint"]) / "e0-checkpoint.json").read_text()))
        if json.loads((root / "protocol.json").read_text())["policy_id"] != policy:
            raise ValueError("confirmation checkpoint differs from selection")
        rows = [json.loads(p.read_text()) for p in sorted((root / "attempts").glob("*.json"))]
        summary = summarize_attempts(expected, rows)
        if not summary["complete"] or any(r["policy_id"] != policy for r in rows):
            raise ValueError("actor confirmation is incomplete or has mixed checkpoints")
        ledgers[name] = {r["episode_uid"]: r for r in rows}
        tensor = np.zeros((10, 50, 3), np.float64)
        for r in rows:
            tensor[r["task_id"], r["initial_state_id"], r["group"]] = r["success"]
        matrices[name] = tensor
        models[name] = dict(
            summary,
            updates=selection["updates"],
            checkpoint=selection["checkpoint"],
            group_success=tensor.mean(axis=(0, 1)).tolist(),
            policy_failures=sum(r["status"] == "policy_failure" for r in rows),
        )
    rng = np.random.default_rng(42)
    draws = rng.integers(0, 50, (10, 10000, 50))
    comparisons = {}
    for left, right in PAIRS:
        a, b = matrices[left], matrices[right]
        comparisons[left + "-" + right] = dict(
            interval(a - b, draws),
            only_left=int(((a == 1) & (b == 0)).sum()),
            only_right=int(((a == 0) & (b == 1)).sum()),
            both_success=int(((a == 1) & (b == 1)).sum()),
            both_failure=int(((a == 0) & (b == 0)).sum()),
        )
    for name in models:
        models[name]["cluster_interval"] = interval(matrices[name], draws)
    result = dict(
        models=models,
        pairs=comparisons,
        training_seed=42,
        training_repeats=1,
        evaluation_groups=3,
        bootstrap="10000 matched initial-state clusters within each task",
        confirmation_manifest_id=content_id(expected),
    )
    bc = comparisons["B-C"]["ci95"]
    winner = "B" if bc[0] > 0 else "C" if bc[1] < 0 else None
    if all(comparisons[k + "-SFT"]["ci95"][1] < 0 for k in ("B", "C")):
        result["decision"] = "两组均明确低于 SFT，先研判共同 actor 设置与数据问题。"
    elif winner and comparisons[winner + "-SFT"]["ci95"][1] >= 0:
        result["decision"] = winner + " 在当前训练种子的配对评测中胜出，优先进入后续多训练种子验证。"
    else:
        result["decision"] = "当前证据不足以确定后续采用的方案，返回共同研判。"
    write_json(dest / "results.json", result)
    with (dest / "episodes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode_uid",
                "task",
                "initial_state",
                "group",
                "B_success",
                "C_success",
                "SFT_success",
                "B_steps",
                "C_steps",
                "SFT_steps",
            ]
        )
        for row in expected:
            uid = row["episode_uid"]
            writer.writerow(
                [
                    uid,
                    row["task_id"],
                    row["initial_state_id"],
                    row["group"],
                    *[int(ledgers[n][uid]["success"]) for n in models],
                    *[ledgers[n][uid]["policy_steps"] for n in models],
                ]
            )
    training = {}
    streams = []
    for name in ("B", "C"):
        spec = json.loads((work / name / "training.json").read_text())
        metrics = read_jsonl(Path(spec["run_dir"]) / "metrics.jsonl")
        if len(metrics) != 60000 or [r["updates"] for r in metrics] != list(range(1, 60001)):
            raise ValueError("actor training updates incomplete")
        streams.append(metrics)
        training[name] = dict(
            exposure={
                k: sum(r["exposure"][k] for r in metrics)
                for k in ("rollout_frames", "positive_frames", "negative_frames", "positive_rollout_frames")
            },
            task_frame_counts=np.sum([r["exposure"]["task_frame_counts"] for r in metrics], axis=0).tolist(),
            curves=[{k: r[k] for k in ("updates", "loss", "grad_norm")} for r in metrics[::100]],
        )
    for a, b in zip(*streams, strict=True):
        if a["rng_key"] != b["rng_key"] or any(
            a["exposure"][k] != b["exposure"][k]
            for k in ("task_frame_counts", "rollout_frames", "observation_action_sha256")
        ):
            raise ValueError("paired actor RNG, sampling or observations/actions differ")
    write_json(dest / "training-audit.json", dict(updates_per_model=60000, paired_inputs_equal=True, models=training))
    cases, media = [], {}
    for left, right in PAIRS:
        for tag, a, b in (("左成功右失败", True, False), ("右成功左失败", False, True), ("双方失败", False, False)):
            chosen = [
                uid
                for uid in sorted(ledgers[left])
                if ledgers[left][uid]["success"] == a and ledgers[right][uid]["success"] == b
            ][:2]
            for uid in chosen:
                clips = []
                for name in (left, right):
                    key = (name, uid)
                    if key not in media:
                        attempt = ledgers[name][uid]
                        media[key] = None
                        if attempt.get("trajectory") and attempt["policy_steps"]:
                            filename = f"case-{len(media):03d}.mp4"
                            video(attempt, dest / filename)
                            media[key] = filename
                    clips.append((name, media[key]))
                cases.append(dict(pair=left + "-" + right, outcome=tag, episode_uid=uid, videos=clips))
    write_json(dest / "cases.json", cases)

    def table(headers, rows):
        return (
            "<table><tr>"
            + "".join("<th>" + html.escape(str(s)) + "</th>" for s in headers)
            + "</tr>"
            + "".join("<tr>" + "".join("<td>" + html.escape(str(s)) + "</td>" for s in r) + "</tr>" for r in rows)
            + "</table>"
        )

    def pct(v):
        return f"{100*v:.2f}%"

    page = [
        '<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">',
        "<title>B/C actor 配对评审</title><style>body{font:16px/1.65 system-ui;max-width:1180px;margin:40px auto;padding:20px}td,th{padding:9px;border-bottom:1px solid #ddd}table{border-collapse:collapse}video{width:48%}pre{white-space:pre-wrap}a{color:#135db3}</style>",
        "<h1>B-clip／C-clip 冻结 critic 的 R1 actor 对照</h1><p>" + result["decision"] + "</p>",
        "<p>两个 actor 均从 pi05_base 训练 60k 步，seed 42。按独立开发集选 checkpoint，再在共同的三组评测种子上比较。区间来自任务内初始状态簇的配对重采样。</p>",
        table(
            ["策略", "选中 EMA", "任务宏平均", "三组评测成功率"],
            [
                [n, r["updates"], pct(r["macro_success"]), ", ".join(map(pct, r["group_success"]))]
                for n, r in models.items()
            ],
        ),
        "<h2>配对差值</h2>",
        table(
            ["配对", "差值", "95% 区间", "仅左成功", "仅右成功"],
            [
                [k, pct(r["mean"]), str([pct(v) for v in r["ci95"]]), r["only_left"], r["only_right"]]
                for k, r in comparisons.items()
            ],
        ),
        "<h2>逐任务结果</h2>",
        table(
            ["Long task", "B", "C", "SFT", "B−C"],
            [
                [t, *[pct(matrices[n][t].mean()) for n in models], pct((matrices["B"][t] - matrices["C"][t]).mean())]
                for t in range(10)
            ],
        ),
        "<h2>共同预算下的开发结果</h2>",
        table(
            ["更新数", "B", "C"],
            [
                [
                    step,
                    *[
                        pct(next(c["macro_success"] for c in selections[n]["candidates"] if c["updates"] == step))
                        for n in ("B", "C")
                    ],
                ]
                for step in (30000, 45000, 60000)
            ],
        ),
        '<p><a href="results.json">完整结果与区间</a> · <a href="episodes.csv">逐 episode 配对</a> · <a href="training-audit.json">训练曝光和输入一致性</a></p>',
        "<h2>同一初始状态与随机种子的执行案例</h2><p>每个配对、每个结果类别按 episode ID 顺序取前两例。</p>",
    ]
    for case in cases:
        page.append(
            "<h3>"
            + html.escape(case["pair"] + " · " + case["outcome"])
            + "</h3><p>"
            + html.escape(case["episode_uid"])
            + "</p>"
        )
        for name, filename in case["videos"]:
            page.append(
                f'<p>{name}</p><video src="{filename}" controls muted preload="metadata"></video>'
                if filename
                else f"<p>{name} 无合法动作视频，详见 episode 记录。</p>"
            )
    page.append("<p>本轮包含一个训练种子和三组评测种子。结果用于筛选后续方案，跨训练种子的复现由后续实验检验。</p>")
    (dest / "index.html").write_text("".join(page), encoding="utf8")
