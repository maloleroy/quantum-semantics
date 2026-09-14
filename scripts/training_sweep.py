"""150 configurations, 150 epochs per fit, five-fold CV plus a development refit."""

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
    profiles = [(subset, "dedupe", "uniform", None) for subset in SUBSETS]
    profiles += [
        (
            subset,
            "strict",
            "uniform" if len(subset) == 1 else "balanced",
            None if len(subset) == 1 else 5000,
        )
        for subset in SUBSETS
    ]
    profiles += [(SUBSETS[-1], "basic", "uniform", 5000)]
    result = []
    for objective, profile, setting in itertools.product(("causal", "cbow"), profiles, SETTINGS):
        datasets, cleaning, sampling, max_sentences = profile
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
                "epochs": 150,
                "samples_per_epoch": 5000,
                "eval_examples": 2048,
                "folds": 5,
                "seed": 42,
                "max_sentences": max_sentences,
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
        if key not in ("experiment_id", "datasets") and value is not None:
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
    parser.add_argument(
        "--max-sentences",
        type=int,
        help="Override the sample size (omit for the full/sample matrix defaults)",
    )
    parser.add_argument("--epochs", type=int, help="Override epochs per fold/refit (default: 150)")
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        help="Override sampled training examples per epoch (default: 5000)",
    )
    parser.add_argument(
        "--eval-examples", type=int, help="Override monitoring examples per split (default: 2048)"
    )
    args = parser.parse_args()
    matrix = experiments()
    if args.list:
        print(json.dumps(matrix, indent=2))
        return
    if args.max_sentences is not None and args.max_sentences < 8:
        parser.error("--max-sentences must be at least 8 for five-fold cross-validation")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    for name in ("samples_per_epoch", "eval_examples"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
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
            job.update(max_sentences=16, epochs=2, samples_per_epoch=32, eval_examples=32)
        if args.max_sentences is not None:
            job["max_sentences"] = args.max_sentences
        if args.epochs is not None:
            job["epochs"] = args.epochs
        for name in ("samples_per_epoch", "eval_examples"):
            if getattr(args, name) is not None:
                job[name] = getattr(args, name)
        print(f"Experiment {job_id:03d}: {json.dumps(job)}", flush=True)
        # A fresh process releases device caches between local smoke runs.
        subprocess.run(
            command(job, args.device, args.output / f"experiment-{job_id:03d}"),
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
