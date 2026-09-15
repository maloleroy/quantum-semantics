"""Run one focused semantic configuration with three independent 64/16/20 repeats."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from qcse.outputs import write_json

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)

# Ten jobs are enough to establish the reference, circuit control, and one-at-a-time
# effects of the settings that were useful in the local check.
EXPERIENCES = (
    {
        "name": "baseline",
        "alpha": 0.05,
        "learning_rate": 0.003,
        "window": 4,
        "layers": 2,
        "batch_size": 64,
        "ansatz": "trainable",
    },
    {
        "name": "no-circuit",
        "alpha": 0.05,
        "learning_rate": 0.003,
        "window": 4,
        "layers": 2,
        "batch_size": 64,
        "ansatz": "zero",
    },
    {
        "name": "lr-low",
        "alpha": 0.05,
        "learning_rate": 0.001,
        "window": 4,
        "layers": 2,
        "batch_size": 64,
        "ansatz": "trainable",
    },
    {
        "name": "lr-high",
        "alpha": 0.05,
        "learning_rate": 0.01,
        "window": 4,
        "layers": 2,
        "batch_size": 64,
        "ansatz": "trainable",
    },
    {
        "name": "alpha-low",
        "alpha": 0.01,
        "learning_rate": 0.003,
        "window": 4,
        "layers": 2,
        "batch_size": 64,
        "ansatz": "trainable",
    },
    {
        "name": "alpha-high",
        "alpha": 0.2,
        "learning_rate": 0.003,
        "window": 4,
        "layers": 2,
        "batch_size": 64,
        "ansatz": "trainable",
    },
    {
        "name": "window-small",
        "alpha": 0.05,
        "learning_rate": 0.003,
        "window": 2,
        "layers": 2,
        "batch_size": 64,
        "ansatz": "trainable",
    },
    {
        "name": "window-large",
        "alpha": 0.05,
        "learning_rate": 0.003,
        "window": 8,
        "layers": 2,
        "batch_size": 64,
        "ansatz": "trainable",
    },
    {
        "name": "layers-four",
        "alpha": 0.05,
        "learning_rate": 0.003,
        "window": 4,
        "layers": 4,
        "batch_size": 64,
        "ansatz": "trainable",
    },
    {
        "name": "batch-128",
        "alpha": 0.05,
        "learning_rate": 0.003,
        "window": 4,
        "layers": 2,
        "batch_size": 128,
        "ansatz": "trainable",
    },
)


def best_validation_row(history):
    """Select an epoch by retrieval top-1, with deterministic secondary keys."""
    return max(
        history,
        key=lambda row: (
            row["validation"]["cosine_top1"],
            row["validation"]["cosine_top5"],
            -row["validation"]["cross_entropy"],
            -row["epoch"],
        ),
    )


def command(config, args, seed, output):
    values = {
        "datasets": ("phrases", "cleaned"),
        "max-sentences": args.max_sentences,
        "epochs": args.epochs,
        "samples-per-epoch": args.samples_per_epoch,
        "eval-examples": args.eval_examples,
        "final-eval-examples": args.final_eval_examples,
        "batch-size": config["batch_size"],
        "window": config["window"],
        "embedding-dim": 16,
        "qubits": 4,
        "layers": config["layers"],
        "alpha": config["alpha"],
        "learning-rate": config["learning_rate"],
        "ansatz": config["ansatz"],
        "seed": seed,
        "threads": args.threads,
        "device": args.device,
        "evaluate-test": True,
        "output": output,
    }
    result = [sys.executable, str(ROOT / "scripts" / "semantic_experiment.py")]
    for key, value in values.items():
        if value is None:
            continue
        result.append(f"--{key}")
        if value is True:
            continue
        if isinstance(value, (tuple, list)):
            result.extend(str(item) for item in value)
        else:
            result.append(str(value))
    return result


def run(args):
    config = EXPERIENCES[args.experience_id]
    experience = args.output / f"experience-{args.experience_id:02d}-{config['name']}"
    experience.mkdir(parents=True, exist_ok=True)
    write_json(
        experience / "manifest.json",
        {
            "protocol": "three independent sentence-grouped 64/16/20 repeats",
            "objective": "causal",
            "datasets": ["phrases", "cleaned"],
            "cleaning": "dedupe",
            "sampling": "5,000 replacement draws per epoch from all eligible training examples",
            "evaluation": (
                f"fixed deterministic validation monitors and up to {args.final_eval_examples} "
                "final test examples per run, without replacement"
                if args.final_eval_examples
                else "fixed deterministic validation monitors and complete final test split"
            ),
            "epochs": args.epochs,
            "seeds": list(SEEDS),
            "config": config,
        },
    )
    runs = []
    for repeat, seed in enumerate(SEEDS):
        repeat_root = experience / f"repeat-{repeat:02d}-seed-{seed}"
        repeat_root.mkdir(parents=True, exist_ok=True)
        before = set(repeat_root.glob("semantic-causal-*"))
        subprocess.run(command(config, args, seed, repeat_root), cwd=ROOT, check=True)
        created = sorted(set(repeat_root.glob("semantic-causal-*")) - before)
        if len(created) != 1:
            raise RuntimeError(f"Expected one run under {repeat_root}, found {created}")
        run_dir = created[0]
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        test = json.loads((run_dir / "test.json").read_text(encoding="utf-8"))
        best = best_validation_row(history)
        runs.append(
            {
                "repeat": repeat,
                "seed": seed,
                "run": str(run_dir),
                "best_epoch": best["epoch"],
                "best_validation": {
                    "cosine_top1": best["validation"]["cosine_top1"],
                    "cosine_top5": best["validation"]["cosine_top5"],
                    "cross_entropy": best["validation"]["cross_entropy"],
                },
                "test": {
                    "cosine_top1": test["cosine_top1"],
                    "cosine_top5": test["cosine_top5"],
                },
            }
        )
    write_json(
        experience / "cv-summary.json",
        {
            "config": config,
            "epochs": args.epochs,
            "eval_examples": args.eval_examples,
            "final_eval_examples": args.final_eval_examples,
            "runs": runs,
        },
    )
    print(f"Completed experience {args.experience_id}: {experience}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="Print the ten-job manifest")
    action.add_argument("--experience-id", type=int, choices=range(len(EXPERIENCES)))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples-per-epoch", type=int, default=5000)
    parser.add_argument("--eval-examples", type=int, default=2048)
    parser.add_argument("--final-eval-examples", type=int, default=10000)
    parser.add_argument(
        "--max-sentences",
        type=int,
        help="Optional local speed cap; omit on the cluster to use all curated sentences",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=Path("outputs/semantic-cluster-50"))
    args = parser.parse_args()
    if args.list:
        print(json.dumps(EXPERIENCES, indent=2))
        return
    if min(args.epochs, args.samples_per_epoch, args.eval_examples, args.threads) < 1:
        parser.error("epochs, samples-per-epoch, eval-examples and threads must be positive")
    if args.final_eval_examples < 0:
        parser.error("final-eval-examples must be nonnegative")
    if args.max_sentences is not None and args.max_sentences < 8:
        parser.error("max-sentences must be at least 8")
    run(args)


if __name__ == "__main__":
    main()
