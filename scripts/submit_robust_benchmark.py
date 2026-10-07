"""Submit independent bounded DGX arrays from an immutable prepared study."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from qcse.benchmark import load_manifest, load_selection
from qcse.outputs import write_json

ROOT = Path(__file__).resolve().parents[1]


def array_commands(study, stage, tasks, concurrency, array_size, options):
    chunks = (tasks + array_size - 1) // array_size
    if concurrency < chunks:
        raise ValueError("Concurrency must cover every independent array chunk")
    limit = concurrency // chunks
    chunk_size = (tasks + chunks - 1) // chunks
    return [
        [
            "sbatch",
            "--parsable",
            f"--array=0-{min(chunk_size, tasks - offset) - 1}%{limit}",
            f"--chdir={ROOT}",
            *options,
            f"--export=ALL,BENCHMARK_STUDY={study},BENCHMARK_STAGE={stage},BENCHMARK_OFFSET={offset}",
            str(ROOT / "slurm-robust-prod10.sbatch"),
        ]
        for offset in range(0, tasks, chunk_size)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--stage", choices=("fit", "refit"), default="fit")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--array-size",
        type=int,
        default=1000,
        help="Maximum tasks per array; adapt to the site's MaxArraySize",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sbatch-option",
        action="append",
        default=[],
        help="Repeat with --sbatch-option=--time=24:00:00, etc.",
    )
    args = parser.parse_args()
    study = args.study.resolve()
    if "," in str(study) or "\n" in str(study) or min(args.concurrency, args.array_size) < 1:
        parser.error("Invalid study path/concurrency")
    manifest = load_manifest(study)
    tasks = manifest["tasks"] if args.stage == "fit" else load_selection(study)["refits"]
    resources = {
        "--partition",
        "--gres",
        "--time",
        "--cpus-per-task",
        "--mem",
        "--account",
        "--qos",
        "--constraint",
    }
    if any(option.split("=")[0] not in resources for option in args.sbatch_option):
        parser.error(
            "Only resource options may be overridden; arrays/export/dependencies are fixed"
        )
    try:
        commands = array_commands(
            study, args.stage, len(tasks), args.concurrency, args.array_size, args.sbatch_option
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(dict(tasks=len(tasks), commands=commands), indent=2), flush=True)
    if args.dry_run:
        return
    (ROOT / "logs").mkdir(exist_ok=True)
    for command in commands:
        result = subprocess.check_output(command, cwd=ROOT, text=True).strip()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        write_json(
            study / "submissions" / f"{args.stage}-{stamp}.json",
            dict(
                job=result,
                stage=args.stage,
                command=command,
                manifest=manifest["fingerprint"],
                submitted_at=stamp,
            ),
        )
        print(f"Submitted {args.stage} array {result}", flush=True)


if __name__ == "__main__":
    main()
