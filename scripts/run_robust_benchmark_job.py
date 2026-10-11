"""Run allocation checks and one fit, recording each subprocess's wall time."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from qcse.outputs import write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("pilot", "fit", "refit"))
    cli.add_argument("study", type=Path)
    cli.add_argument("--task-id", type=int, required=True)
    args = cli.parse_args()
    if args.task_id < 0:
        cli.error("Task ID must be nonnegative")
    stage = "fit" if args.stage == "pilot" else args.stage
    output = args.study / ("cv" if stage == "fit" else "refit") / f"task-{args.task_id:04d}"
    scheduler = {
        k: os.environ.get(k)
        for k in (
            "SLURM_JOB_ID",
            "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "CUDA_VISIBLE_DEVICES",
        )
    }
    array = scheduler["SLURM_ARRAY_JOB_ID"]
    job = f"{array}_{scheduler['SLURM_ARRAY_TASK_ID']}" if array else scheduler["SLURM_JOB_ID"]
    ledger: dict = dict(
        schema=1,
        stage=args.stage,
        task_id=args.task_id,
        scheduler=scheduler,
        job=job,
        started_at=datetime.now(UTC).isoformat(),
        status="running",
        reused_result=(output / "result.json").exists(),
        phases=[],
    )
    path = args.study / "job-timings" / f"{job or 'local'}-{uuid4().hex}.json"
    started = time.monotonic()

    def run(name, command):
        row: dict = dict(name=name, command=command, status="running", seconds=None)
        ledger["phases"].append(row)
        write_json(path, ledger)
        begun = time.monotonic()
        try:
            subprocess.run(command, check=True)
        except BaseException as error:
            row.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=type(error).__name__,
            )
            if isinstance(error, subprocess.CalledProcessError):
                row["returncode"] = error.returncode
            raise
        else:
            row.update(status="complete", returncode=0)
        finally:
            row["seconds"] = time.monotonic() - begun
            write_json(path, ledger)

    python = sys.executable
    try:
        run("device_inventory", ["nvidia-smi"])
        run("backend_preflight", [python, "scripts/check_backend.py", "--device", "cuda"])
        run(
            "adapter_preflight",
            [python, "scripts/check_benchmark_backend.py", "--device", "cuda", "--trainability"],
        )
        if args.stage == "pilot":
            manifest = json.loads((args.study / "manifest.json").read_text())
            objective = manifest["tasks"][args.task_id]["objective"]
            profile = str(args.study / f"profile-{objective}.json")
            run(
                "profile",
                [
                    python,
                    "scripts/profile_robust_benchmark.py",
                    str(args.study),
                    "--device",
                    "cuda",
                    "--objective",
                    objective,
                    "--output",
                    profile,
                ],
            )
            run(
                "estimate",
                [
                    python,
                    "scripts/estimate_benchmark_compute.py",
                    str(args.study),
                    profile,
                    "--concurrency",
                    "4",
                    "--output",
                    str(args.study / "compute.json"),
                ],
            )
            if any("qubits" in task["arm"] for task in manifest["tasks"]):
                run(
                    "trainability_probe",
                    [
                        python,
                        "scripts/probe_barren_plateaus.py",
                        "--output",
                        str(args.study / "trainability-probe"),
                        "--device",
                        "cuda",
                        "--seeds",
                        "32",
                    ],
                )
        run(
            "fit",
            [
                python,
                "scripts/robust_benchmark.py",
                stage,
                str(args.study),
                "--task-id",
                str(args.task_id),
                "--device",
                "cuda",
                "--resume",
            ],
        )
    except BaseException as error:
        ledger.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=type(error).__name__,
        )
        raise
    else:
        ledger["status"] = "complete"
        print("Benchmark job completed successfully.", flush=True)
    finally:
        ledger.update(seconds=time.monotonic() - started, finished_at=datetime.now(UTC).isoformat())
        write_json(path, ledger)


if __name__ == "__main__":
    main()
