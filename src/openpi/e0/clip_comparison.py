"""Comparisons in common reward units, preserving episode pairing."""

import csv
import html
from pathlib import Path

import numpy as np
from openpi_client.e0.artifacts import write_json
from openpi.e0.critic_report import estimates
from openpi.e0.reward_study import CLIP_METHODS, MODES


def strict_pairs():
    return [(f"{method.removesuffix('_clip')}-{mode}", f"{method}-{mode}") for method in CLIP_METHODS for mode in MODES]


def compare(dest, frames, attempts, summary):
    measures, terminal_rows = {}, []
    for old, new in strict_pairs():
        shared = {}
        for uid, baseline in frames[old].items():
            clipped = frames[new][uid]
            raw = clipped["before"]
            np.testing.assert_array_equal(raw, baseline["before"])
            bounded = np.clip(raw, -1, 0)
            row = {}
            for region, subset in (("all", np.ones(len(raw), bool)), ("below", raw < -1), ("within", raw >= -1)):
                if not subset.any():
                    continue
                for label, target in (("return", raw), ("bounded", bounded)):
                    key = f"{region}_{label}_mae"
                    for variant, prediction in (("scaled", baseline), ("clip", clipped)):
                        row[f"{key}_{variant}"] = float(np.abs(prediction["value"][subset] - target[subset]).mean())
                    row[key + "_delta"] = row[key + "_clip"] - row[key + "_scaled"]
            shared[uid] = row
            if attempts[uid]["success"] or attempts[uid]["policy_steps"] != 520:
                continue
            last = dict(method=old, clip_method=new, episode_uid=uid, task=attempts[uid]["task_id"])
            for variant, model in (("scaled", baseline), ("clip", clipped)):
                flag = model["positive"]
                last.update(
                    {
                        variant + "_positive": int(flag[-1]),
                        variant + "_last50_fraction": float(flag[-50:].mean()),
                        variant + "_value": float(model["value"][-1]),
                        variant + "_advantage": float(model["advantage"][-1]),
                        variant + "_threshold": float(model["threshold"]),
                    }
                )
                np.testing.assert_allclose(model["advantage"][-1], -0.5 - float(model["value"][-1]), atol=1e-6)
            last["transition"] = f"{last['scaled_positive']}->{last['clip_positive']}"
            terminal_rows.append(last)
        terminal = [r for r in terminal_rows if r["method"] == old]
        measures[old] = dict(
            common_errors=estimates(shared, attempts),
            transitions={k: sum(r["transition"] == k for r in terminal) for k in ("0->0", "0->1", "1->0", "1->1")},
            terminal50=estimates(
                {
                    r["episode_uid"]: dict(
                        scaled=r["scaled_last50_fraction"],
                        clip=r["clip_last50_fraction"],
                        difference=r["clip_last50_fraction"] - r["scaled_last50_fraction"],
                    )
                    for r in terminal
                },
                attempts,
            ),
        )
    write_json(dest / "common-reference.json", measures)
    if terminal_rows:
        with (dest / "terminal-pairs.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(terminal_rows[0]))
            writer.writeheader()
            writer.writerows(terminal_rows)
    blocks = [
        "<meta charset=utf-8><title>Clip 配对评审</title><style>body{font:16px system-ui;max-width:1250px;margin:36px auto}td,th{border-bottom:1px solid #ddd;padding:6px}details{margin-bottom:12px}summary{cursor:pointer}</style>",
        "<h1>B/C 直接 clip 配对评审</h1><p>相同轨迹上的配对结果。误差和翻转率先按轨迹，再按任务平均。区间为按任务重采样轨迹的 95% 区间。误差差值取 clip 减原缩放。</p>",
        '<p><a href="index.html">十个模型与轨迹图册</a> · <a href="clip-disagreements.html">clip 分歧片段</a> · <a href="common-reference.json">共同参照误差及区间</a> · <a href="study.json">逐任务与分层统计</a> · <a href="terminal-pairs.csv">超时轨迹逐条明细</a></p>',
    ]

    def table(headers, rows):
        return (
            "<table><tr>"
            + "".join("<th>" + html.escape(h) + "</th>" for h in headers)
            + "</tr>"
            + "".join("<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in row) + "</tr>" for row in rows)
            + "</table>"
        )

    def cell(stats, key):
        v = stats.get(key)
        return "—" if not v else f'{v["mean"]:.4f} [{v["ci95"][0]:.4f}, {v["ci95"][1]:.4f}]'

    for old, new in strict_pairs():
        group = measures[old]
        stats = summary["pairs"][old + " vs " + new]
        blocks.append("<h2>" + html.escape(old) + " → clip</h2>")
        m = stats["macro"]
        tr = group["transitions"]
        blocks.append(
            "<p>翻转率 "
            + cell(m, "flip")
            + " · Spearman "
            + cell(m, "spearman")
            + " · Jaccard "
            + cell(m, "jaccard")
            + "</p>"
        )
        blocks.append(
            "<p>超时末帧：正转负 "
            + str(tr["1->0"])
            + "，负转正 "
            + str(tr["0->1"])
            + "，保持正 "
            + str(tr["1->1"])
            + "，保持负 "
            + str(tr["0->0"])
            + "。</p>"
        )
        for task, values in [("宏平均", group["common_errors"]["macro"]), *group["common_errors"]["by_task"].items()]:
            blocks.append(
                "<details"
                + (" open" if task == "宏平均" else "")
                + "><summary>"
                + html.escape(task)
                + "</summary>"
                + table(
                    ["帧范围", "共同参照", "原缩放 MAE [CI]", "直接 clip MAE [CI]", "配对差 [CI]"],
                    [
                        [
                            region,
                            label,
                            *[cell(values, f"{key}_{label}_mae_{variant}") for variant in ["scaled", "clip", "delta"]],
                        ]
                        for region, key in [("全部", "all"), ("回报低于 −1", "below"), ("回报至少 −1", "within")]
                        for label in ["return", "bounded"]
                    ],
                )
                + "</details>"
            )
    (dest / "comparison.html").write_text("".join(blocks), encoding="utf8")
