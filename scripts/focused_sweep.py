"""Manifest and launcher for the compact two-repetition benchmark."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CAUSAL_CONFIGS = (
    {"name": "baseline", "alpha": 1.0, "layers": 8, "window": 4, "batch_size": 64},
    {"name": "alpha-0.1", "alpha": 0.1, "layers": 8, "window": 4, "batch_size": 64},
    {"name": "alpha-3", "alpha": 3.0, "layers": 8, "window": 4, "batch_size": 64},
    {"name": "layers-2", "alpha": 1.0, "layers": 2, "window": 4, "batch_size": 64},
    {"name": "window-2", "alpha": 1.0, "layers": 8, "window": 2, "batch_size": 64},
    {"name": "window-8", "alpha": 1.0, "layers": 8, "window": 8, "batch_size": 64},
    {"name": "batch-16", "alpha": 1.0, "layers": 8, "window": 4, "batch_size": 16},
    {"name": "batch-256", "alpha": 1.0, "layers": 8, "window": 4, "batch_size": 256},
)

ENCODING_CONFIGS = (
    {
        "name": "classical-encoding-trainable-ansatz",
        "pathway": "quantum",
        "ansatz": "trainable",
    },
    {
        "name": "classical-encoding-no-ansatz",
        "pathway": "quantum",
        "ansatz": "zero",
    },
    {
        "name": "no-encoding-no-ansatz",
        "pathway": "none",
        "ansatz": "zero",
    },
)

ATTENTION_CONFIGS = (
    {"name": "qcse-encoding-quantum-attention", "encoding": "qcse", "quantum_attention": True},
    {
        "name": "classical-encoding-quantum-attention",
        "encoding": "classical",
        "quantum_attention": True,
    },
    {
        "name": "qcse-encoding-no-quantum-attention",
        "encoding": "qcse",
        "quantum_attention": False,
    },
    {
        "name": "classical-encoding-no-quantum-attention",
        "encoding": "classical",
        "quantum_attention": False,
    },
)

CONFIGS = {
    "causal": CAUSAL_CONFIGS,
    "encoding": ENCODING_CONFIGS,
    "attention": ATTENTION_CONFIGS,
}


def task_configuration(family, task_id):
    configurations = CONFIGS[family]
    if task_id < 0 or task_id >= 2 * len(configurations):
        raise ValueError(f"{family} task ID must lie between 0 and {2 * len(configurations) - 1}")
    configuration_id, repetition = divmod(task_id, 2)
    return configurations[configuration_id], repetition, 42 + repetition


def command(family, task_id, args):
    configuration, repetition, seed = task_configuration(family, task_id)
    output = args.output / family / configuration["name"] / f"repetition-{repetition}"
    common = [
        "--epochs",
        str(args.epochs),
        "--samples-per-epoch",
        str(args.samples_per_epoch),
        "--eval-examples",
        str(args.eval_examples),
        "--final-eval-examples",
        str(args.final_eval_examples),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--min-epochs",
        str(args.min_epochs),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--output",
        str(output),
    ]
    if family == "causal":
        return [
            sys.executable,
            str(ROOT / "scripts" / "causal_focused_experiment.py"),
            "--name",
            configuration["name"],
            "--alpha",
            str(configuration["alpha"]),
            "--layers",
            str(configuration["layers"]),
            "--window",
            str(configuration["window"]),
            "--batch-size",
            str(configuration["batch_size"]),
            "--repetition",
            str(repetition),
            *common,
        ]
    if family == "encoding":
        return [
            sys.executable,
            str(ROOT / "scripts" / "semantic_experiment.py"),
            "--model",
            "semantic",
            "--datasets",
            "phrases",
            "cleaned",
            "--window",
            "4",
            "--embedding-dim",
            "16",
            "--qubits",
            "4",
            "--layers",
            "2",
            "--alpha",
            "0.05",
            "--pathway",
            configuration["pathway"],
            "--ansatz",
            configuration["ansatz"],
            "--threads",
            str(args.threads),
            "--evaluate-test",
            *common,
        ]
    attention_flag = (
        "--quantum-attention" if configuration["quantum_attention"] else "--no-quantum-attention"
    )
    return [
        sys.executable,
        str(ROOT / "scripts" / "semantic_experiment.py"),
        "--model",
        "attention",
        "--encoding",
        configuration["encoding"],
        attention_flag,
        "--window",
        "4",
        "--embedding-dim",
        "16",
        "--qubits",
        "4",
        "--layers",
        "2",
        "--alpha",
        "0.05",
        "--threads",
        str(args.threads),
        "--evaluate-test",
        *common,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(CONFIGS))
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples-per-epoch", type=int, default=5000)
    parser.add_argument("--eval-examples", type=int, default=2048)
    parser.add_argument("--final-eval-examples", type=int, default=10000)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Numerical/PyTorch thread target inside each process",
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/focused-sweep-50"))
    args = parser.parse_args()
    if args.list:
        print(json.dumps(CONFIGS, indent=2))
        return
    if args.family is None or args.task_id is None:
        parser.error("--family and --task-id are required unless --list is used")
    if (
        min(
            args.epochs,
            args.samples_per_epoch,
            args.eval_examples,
            args.early_stopping_patience,
            args.threads,
        )
        < 1
        or args.min_epochs < 0
    ):
        parser.error("budgets and patience must be positive; min-epochs must be nonnegative")
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = str(args.threads)
    subprocess.run(command(args.family, args.task_id, args), check=True, cwd=ROOT, env=environment)


if __name__ == "__main__":
    main()
