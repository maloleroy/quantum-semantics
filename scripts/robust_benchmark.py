"""Prepare/run/select/report the reproducible QCSE versus BERT benchmark."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from qcse.benchmark import (
    configurations,
    fit,
    load_manifest,
    load_selection,
    prepare,
    selection,
)
from qcse.outputs import new_run_directory, run_status, write_json


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("list").add_argument("--trainability", action="store_true")
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, default=Path("outputs/robust-benchmark"))
    prep.add_argument(
        "--objectives", choices=("causal", "cbow"), nargs="+", default=["causal", "cbow"]
    )
    prep.add_argument("--arms", nargs="+", help="Named subset for protocol smoke runs")
    prep.add_argument(
        "--trainability", action="store_true", help="Include qubit/depth/init controls"
    )
    prep.add_argument("--max-sentences", type=int)
    prep.add_argument("--window", type=int, default=4)
    prep.add_argument("--folds", type=int, default=5)
    prep.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    prep.add_argument("--split-seed", type=int, default=2026)
    prep.add_argument("--epochs", type=int, default=50)
    prep.add_argument("--samples-per-epoch", type=int, default=5000)
    prep.add_argument("--batch-size", type=int, default=64)
    prep.add_argument("--microbatch-size", type=int, default=8)
    prep.add_argument("--eval-examples", type=int, default=2048)
    prep.add_argument(
        "--final-eval-sentences",
        type=int,
        default=2000,
        help="Complete sentence groups per final split; 0 means exhaustive",
    )
    prep.add_argument("--patience", type=int, default=8)
    prep.add_argument("--min-epochs", type=int, default=10)
    prep.add_argument("--threads", type=int, default=4)
    for name in ("fit", "refit"):
        run = commands.add_parser(name)
        run.add_argument("study", type=Path)
        run.add_argument("--task-id", type=int, required=True)
        run.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
        run.add_argument("--resume", action="store_true")
    for name in ("select", "status", "report"):
        run = commands.add_parser(name)
        run.add_argument("study", type=Path)
        if name == "report":
            run.add_argument("--bootstrap-draws", type=int, default=2000)
    return cli


def main():
    cli = parser()
    args = cli.parse_args()
    if args.command == "list":
        print(json.dumps(configurations(args.trainability), indent=2))
        return
    if args.command == "prepare":
        settings = vars(args).copy()
        settings.pop("output")
        settings.pop("command")
        if (
            min(
                args.epochs,
                args.samples_per_epoch,
                args.batch_size,
                args.microbatch_size,
                args.eval_examples,
                args.patience,
                args.threads,
                args.window,
            )
            < 1
            or args.folds < 2
            or args.min_epochs < 0
            or args.final_eval_sentences < 0
        ):
            cli.error("Invalid dimensions/budget")
        if "cbow" in args.objectives and args.window % 2:
            cli.error("Bidirectional window must be even")
        if len(set(args.seeds)) != len(args.seeds) or len(set(args.objectives)) != len(
            args.objectives
        ):
            cli.error("Seeds/objectives must be distinct")
        if any(not 0 <= seed < 2**32 for seed in args.seeds):
            cli.error("Seeds must lie in 0..2**32-1")
        if args.arms and not set(args.arms) <= {
            a["name"] for a in configurations(args.trainability)
        }:
            cli.error("Unknown arm; use list to inspect names")
        with new_run_directory(args.output, "qcse-bert", dict(stage="prepare")) as root:
            manifest = prepare(root, settings)
            print(f"Prepared {len(manifest['tasks'])} CV tasks at {root}")
        return
    root = args.study.resolve()
    manifest = load_manifest(root)
    if args.command == "status":
        complete = [
            i
            for i in range(len(manifest["tasks"]))
            if (root / "cv" / f"task-{i:04d}" / "result.json").exists()
        ]
        print(
            json.dumps(
                dict(
                    study=str(root),
                    complete=len(complete),
                    total=len(manifest["tasks"]),
                    missing=[i for i in range(len(manifest["tasks"])) if i not in complete],
                )
            )
        )
    elif args.command == "select":
        result = selection(root)
        print(f"Frozen validation-only selection; {len(result['refits'])} refit tasks")
    elif args.command == "report":
        from qcse.benchmark_report import report

        report(root, args.bootstrap_draws)
    else:
        tasks = manifest["tasks"] if args.command == "fit" else load_selection(root)["refits"]
        if not 0 <= args.task_id < len(tasks):
            cli.error(f"Task ID must lie in 0..{len(tasks) - 1}")
        task = tasks[args.task_id]
        output = root / ("cv" if args.command == "fit" else "refit") / f"task-{args.task_id:04d}"
        attempt = output / "attempts" / f"{uuid4().hex}.json"
        try:
            with run_status(output, dict(stage=args.command, task=task)):
                fit(
                    root,
                    task,
                    output,
                    args.device,
                    args.resume,
                    None if args.command == "fit" else task["epochs"],
                )
        finally:
            write_json(attempt, json.loads((output / "run_info.json").read_text()))


if __name__ == "__main__":
    main()
