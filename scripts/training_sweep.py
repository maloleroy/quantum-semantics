"""150 ten-epoch experiments, grouped into 30 jobs of five sequential runs."""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBSETS = [
    list(group)
    for count in (1, 2, 3)
    for group in itertools.combinations(("phrases", "tatoeba", "cleaned"), count)
]
# Match batch sizes at 2 and 8 layers, then stress the deeper circuit.
SETTINGS = [(2, 16), (2, 64), (8, 16), (8, 64), (64, 256)]


def experiments():
    profiles = [(subset, "dedupe", "uniform") for subset in SUBSETS]
    profiles += [(subset, "strict", "balanced") for subset in SUBSETS]
    profiles += [(SUBSETS[-1], "basic", "uniform")]
    result = []
    for objective, profile, setting in itertools.product(("causal", "cbow"), profiles, SETTINGS):
        datasets, cleaning, sampling = profile
        layers, batch_size = setting
        result.append(
            {
                "experiment_id": len(result),
                "objective": objective,
                "datasets": datasets,
                "cleaning": cleaning,
                "sampling": sampling,
                "layers": layers,
                "batch_size": batch_size,
                "simulation_batch_size": batch_size,
                "epochs": 10,
                "seed": 42,
                "max_sentences": 128,
                "max_examples": 512,
            }
        )
    return result


def command(job, device, output):
    args = [
        sys.executable,
        "-m",
        "qcse.cli",
        "train",
        "--device",
        device,
        "--output",
        str(output),
        "--datasets",
        *job["datasets"],
    ]
    for key, value in job.items():
        if key not in ("experiment_id", "datasets"):
            args.extend(["--" + key.replace("_", "-"), str(value)])
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--list", action="store_true", help="Print the reproducible 150-experiment manifest"
    )
    action.add_argument("--experiment-id", type=int, choices=range(150), metavar="0..149")
    action.add_argument(
        "--group-id",
        type=int,
        choices=range(30),
        metavar="0..29",
        help="Run five consecutive experiments in one Slurm job",
    )
    action.add_argument(
        "--smoke", action="store_true", help="16 small runs covering both objectives"
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "sweep")
    parser.add_argument("--max-examples", type=int, help="Override the per-experiment example cap")
    args = parser.parse_args()
    matrix = experiments()
    if args.list:
        print(json.dumps(matrix, indent=2))
        return
    if args.max_examples is not None and args.max_examples < 2:
        parser.error("--max-examples must be at least 2")
    if args.smoke:
        # All seven source subsets, all cleaning modes, both sampling modes,
        # and all five layer/batch settings, with the same full vocabulary.
        profile_ids = [0, 1, 2, 10, 11, 12, 13, 14]
        ids = [
            objective * 75 + profile * 5 + i % len(SETTINGS)
            for objective in range(2)
            for i, profile in enumerate(profile_ids)
        ]
    elif args.group_id is not None:
        ids = list(range(args.group_id * 5, args.group_id * 5 + 5))
    else:
        ids = [args.experiment_id]
    for job_id in ids:
        job = matrix[job_id].copy()
        if args.smoke:
            job.update(max_sentences=16, max_examples=64)
        if args.max_examples is not None:
            job["max_examples"] = args.max_examples
        print(f"Experiment {job_id:03d}: {json.dumps(job)}", flush=True)
        # A fresh process releases device caches between local smoke runs.
        subprocess.run(
            command(job, args.device, args.output / f"experiment-{job_id:03d}"),
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
