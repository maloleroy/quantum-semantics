"""Submit independent bounded DGX arrays from an immutable prepared study."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from qcse.benchmark import load_manifest, load_selection
from qcse.outputs import write_json

ROOT = Path(__file__).resolve().parents[1]
MAX_CONCURRENCY = 4
# Other states may hold resources or return to the queue. Unknown states block.
FINISHED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "TIMEOUT",
}


def array_commands(study, stage, tasks, concurrency, array_size, options):
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError("Concurrency must be between 1 and 4")
    chunks = (tasks + array_size - 1) // array_size
    if concurrency < chunks:
        raise ValueError("Concurrency must cover every independent array chunk")
    limit = concurrency // chunks
    chunk_size = (tasks + chunks - 1) // chunks
    return [
        [
            "sbatch",
            "--parsable",
            "--partition=prod10",
            "--gres=gpu:nvidia_a100_1g.10gb:1",
            "--nodes=1",
            "--ntasks=1",
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
    parser.add_argument("--stage", choices=("pilot", "fit", "refit"), default="fit")
    parser.add_argument(
        "--concurrency", type=int, choices=range(1, MAX_CONCURRENCY + 1), default=MAX_CONCURRENCY
    )
    parser.add_argument(
        "--array-size",
        type=int,
        default=1000,
        help="Maximum tasks per array; adapt to the site's MaxArraySize",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--wait", action="store_true", help="Wait for each sbatch array; fail if a task fails"
    )
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
    if args.stage == "refit":
        tasks = load_selection(study)["refits"]
    elif args.stage == "pilot":
        tasks = manifest["tasks"][:1]
    else:
        tasks = manifest["tasks"]
    resources = {
        "--time",
        "--cpus-per-task",
        "--mem",
        "--account",
        "--qos",
        "--constraint",
    }
    if any(option.split("=")[0] not in resources for option in args.sbatch_option):
        parser.error(
            "Only time/CPU/memory/account/QoS/constraint options may be overridden; "
            "the prod10 partition, one 10 GB MIG per job and protocol controls are fixed"
        )
    try:
        commands = array_commands(
            study,
            args.stage,
            len(tasks),
            1 if args.stage == "pilot" else args.concurrency,
            args.array_size,
            args.sbatch_option + (["--wait"] if args.wait else []),
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(dict(tasks=len(tasks), commands=commands), indent=2), flush=True)
    if args.dry_run:
        return
    (ROOT / "logs").mkdir(exist_ok=True)
    with (ROOT / "logs" / "benchmark-submission.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another submission is in progress in this checkout")
        queue = subprocess.check_output(
            [
                "squeue",
                "--me",
                "--partition=prod10",
                "--states=all",
                "--noheader",
                "--format=%i|%T",
            ],
            cwd=ROOT,
            text=True,
        )
        active = [
            row.strip()
            for row in queue.splitlines()
            if row.strip()
            and ("|" not in row or row.rsplit("|", 1)[-1].strip() not in FINISHED_STATES)
        ]
        if active:
            parser.error(
                "Existing prod10 jobs prevent an overlapping submission. "
                f"Wait for them to finish, including any pilot job. Jobs: {', '.join(active)}"
            )
        launcher_source = Path(__file__).read_bytes()
        for command in commands:
            # Inherited SBATCH_* settings must not override the checked resources.
            environment = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            result = subprocess.check_output(command, cwd=ROOT, text=True, env=environment).strip()
            write_json(
                study / "submissions" / f"{args.stage}-{stamp}.json",
                dict(
                    job=result,
                    stage=args.stage,
                    command=command,
                    manifest=manifest["fingerprint"],
                    submitted_at=stamp,
                    wait_completed_at=datetime.now(UTC).isoformat() if args.wait else None,
                    launcher_sha256=hashlib.sha256(launcher_source).hexdigest(),
                ),
            )
            (study / "submissions" / f"launcher-{stamp}.py").write_bytes(launcher_source)
            print(f"Submitted {args.stage} array {result}", flush=True)


if __name__ == "__main__":
    main()
