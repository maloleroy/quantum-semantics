"""45 causal configurations, 50 epochs, 2-fold 64-16-20 CV plus a refit."""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (0.1, 1.0, 3.0)
LEARNING_RATES = (0.0001, 0.0003, 0.001)
WINDOWS = (2, 4, 8)
# Extra settings compare depth at batch 64 and batch size at each depth.
# The shared reference (8 layers, batch 64) is already in the main grid.
SETTINGS = ((2, 16), (2, 64), (8, 16), (8, 256), (64, 64), (64, 256))


def experiments():
    profiles = [
        dict(alpha=alpha, learning_rate=lr, window=window, layers=8, batch_size=64)
        for alpha, lr, window in itertools.product(ALPHAS, LEARNING_RATES, WINDOWS)
    ]
    profiles += [
        dict(alpha=1.0, learning_rate=0.0003, window=window, layers=layers, batch_size=batch)
        for (layers, batch), window in itertools.product(SETTINGS, (2, 8))
    ]
    profiles += [
        dict(
            alpha=1.0,
            learning_rate=0.0003,
            window=4,
            layers=8,
            batch_size=64,
            max_sentences=size,
            sampling=sampling,
        )
        for size, sampling in itertools.product((5000, 20000, 50000), ("uniform", "balanced"))
    ]
    result = []
    for profile in profiles:
        result.append(
            {
                "experiment_id": len(result),
                "objective": "causal",
                "datasets": ["phrases", "cleaned"],
                "cleaning": "dedupe",
                "sampling": "uniform",
                "max_sentences": None,
                **profile,
                "simulation_batch_size": profile["batch_size"],
                "epochs": 50,
                "samples_per_epoch": 5000,
                "eval_examples": 2048,
                "folds": 2,
                "val_fraction": 0.2,
                "seed": 42,
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
        "--list", action="store_true", help="Print the reproducible 45-experiment manifest"
    )
    action.add_argument("--experiment-id", type=int, choices=range(45), metavar="0..44")
    action.add_argument(
        "--group-id",
        type=int,
        choices=range(9),
        metavar="0..8",
        help="Run five consecutive experiments in one Slurm job",
    )
    action.add_argument(
        "--smoke", action="store_true", help="7 small causal runs covering the settings"
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "sweep")
    parser.add_argument(
        "--max-sentences",
        type=int,
        help="Limit the sentence pool for a short check (default: all curated sentences)",
    )
    parser.add_argument("--epochs", type=int, help="Override epochs per fold/refit (default: 50)")
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
        parser.error("--max-sentences must be at least 8 for cross-validation")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    for name in ("samples_per_epoch", "eval_examples"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.smoke:
        # Grid extremes, reference, depth extremes, sampling extremes.
        ids = [0, 13, 26, 27, 38, 39, 44]
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
