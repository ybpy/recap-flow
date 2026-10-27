"""Thin dispatch only: never submits TIONE tasks."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "resume"):
        p = sub.add_parser(name)
        p.add_argument("manifest")
        p.add_argument("--stop-after", type=int)
    p = sub.add_parser("audit")
    p.add_argument("--source", default="/mnt/datasets/libero_demo_pistar")
    p.add_argument("--rlds-root", default="/mnt/datasets/libero-rlds")
    p.add_argument("--output", default="/mnt/recap-flow-runtime/datasets/sft-v1")
    p.add_argument("--report-dir", default="/mnt/recap-flow-runtime/reports")
    p = sub.add_parser("episodes")
    p.add_argument("--namespace", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--suites", nargs="+", default=["libero_10"])
    p.add_argument("--groups", nargs="+", type=int, default=[0])
    p.add_argument("--attempts", type=int, default=50)
    p.add_argument("--purpose", choices=["rollout", "development", "final", "smoke"], required=True)
    p.add_argument("--training-seed", type=int, required=True)
    p = sub.add_parser("rollout")
    p.add_argument("manifest")
    p.add_argument("--output", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--timeout", type=float, default=120)
    p = sub.add_parser("serve")
    p.add_argument("manifest")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("evaluate")
    p.add_argument("--actor-manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episode-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--devices", nargs="+", type=int, default=[0])
    p.add_argument("--workers-per-device", type=int, default=4)
    p.add_argument("--platform", choices=["cyy_dev", "a800"], required=True)
    p.add_argument("--port-base", type=int, default=9000)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--wall-timeout", type=float, default=86400)
    p = sub.add_parser("predict")
    p.add_argument("manifest")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--autonomous-only", action="store_true")
    p = sub.add_parser("convert-rollouts")
    p.add_argument("--episode-manifest", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--demo-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--purpose", choices=["rollout", "development", "final"], required=True)
    p.add_argument("--branch", choices=["shared", "base_reset", "warm_start"], required=True)
    p.add_argument("--round-index", type=int, required=True)
    p.add_argument("--suite")
    p = sub.add_parser("merge")
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("label")
    p.add_argument("--source", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("report")
    p.add_argument("--manifest", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--output", default="/mnt/recap-flow-runtime/reports/preflight-train.json")
    p = sub.add_parser("egl")
    p.add_argument("--output", default="/mnt/recap-flow-runtime/reports/preflight-egl.json")
    p = sub.add_parser("curves")
    p.add_argument("--index-path", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("final-report")
    p.add_argument("--index-path", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("select-actor")
    p.add_argument("--training-manifest", required=True)
    p.add_argument("--evaluations", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("select-critic")
    p.add_argument("--training-manifest", required=True)
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("assets")
    p.add_argument("--output", default="/mnt/recap-flow-runtime/reports/asset-inventory.json")
    for name in ("schema", "schema-libero"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", default="/mnt/recap-flow-runtime/datasets/sft-v1")
        p.add_argument("--output")
    p = sub.add_parser("test")
    p = sub.add_parser("smoke-configs")
    p.add_argument("--output", default="/mnt/recap-flow-runtime/jobs")
    p.add_argument("--platform", choices=["cyy_dev", "a800"], default="cyy_dev")
    p.add_argument("--name", default="e0-smoke")
    p = sub.add_parser("smoke")
    p.add_argument("--platform", choices=["cyy_dev", "a800"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--phase", choices=["all", "preflight", "training", "recovery", "collect", "closed-loop"], default="all"
    )
    p.add_argument("--jobs-index")
    p = sub.add_parser("pilot-sft")
    p.add_argument("--output", required=True)
    p = sub.add_parser("training-config")
    p.add_argument("--request", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("matrix")
    p.add_argument("--output", required=True)
    p = sub.add_parser("task-card")
    p.add_argument("--manifest", required=True)
    p.add_argument("--bindings", required=True)
    p.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command in ("run", "resume"):
        from openpi.e0.train import run

        run(args.pop("manifest"), resume=command == "resume", **args)
    elif command == "audit":
        from openpi.e0.audit import run

        run(**args)
    elif command == "episodes":
        from openpi_client.e0.artifacts import write_jsonl
        from openpi_client.e0.protocol import episodes

        output = args.pop("output")
        write_jsonl(output, episodes(**args))
    elif command == "rollout":
        from openpi_client.e0.rollout import run

        run(args.pop("manifest"), **args)
    elif command == "serve":
        from openpi.e0.inference import serve

        serve(**args)
    elif command == "evaluate":
        from openpi.e0.evaluation import evaluate

        evaluate(**args)
    elif command == "predict":
        from openpi.e0.inference import critic_predictions

        critic_predictions(**args)
    elif command in ("convert-rollouts", "merge", "label"):
        from openpi.e0 import data

        fn = {"convert-rollouts": data.convert_rollouts, "merge": data.merge, "label": data.relabel}[command]
        fn(**args)
    elif command == "report":
        from openpi_client.e0.artifacts import read_jsonl
        from openpi_client.e0.artifacts import write_json
        from openpi_client.e0.protocol import summarize_attempts

        result = summarize_attempts(
            read_jsonl(args["manifest"]),
            [json.loads(p.read_text()) for p in sorted((Path(args["results"]) / "attempts").glob("*.json"))],
        )
        write_json(args["output"], result)
        print(json.dumps(result, indent=2))
        if not result["complete"]:
            sys.exit(2)
    elif command == "preflight":
        from openpi.e0.preflight import train_environment

        train_environment(**args)
    elif command == "egl":
        from openpi_client.e0.preflight import simulator

        simulator(**args)
    elif command == "curves":
        from openpi.e0.curves import curves

        curves(**args)
    elif command == "final-report":
        from openpi.e0.report import final_report

        final_report(**args)
    elif command in ("select-actor", "select-critic"):
        from openpi.e0 import selection

        (selection.actor if command == "select-actor" else selection.critic)(**args)
    elif command == "assets":
        from openpi.e0.assets import inventory

        inventory(**args)
    elif command in ("schema", "schema-libero"):
        from openpi_client.e0.schema_probe import run

        run(**args)
    elif command == "test":
        import pytest

        sys.exit(pytest.main(["tests/e0", "-q"]))
    elif command == "smoke":
        from openpi.e0.smoke import smoke

        smoke(**args)
    elif command in ("smoke-configs", "pilot-sft", "training-config", "freeze", "matrix", "task-card"):
        from openpi.e0 import jobs

        getattr(jobs, command.replace("-", "_"))(**args)


if __name__ == "__main__":
    main()
