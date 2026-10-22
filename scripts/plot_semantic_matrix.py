"""Aggregate cosine metrics and plots for the semantic dataset/setup matrix."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def dataset_name(settings):
    selected = tuple(settings.get("datasets") or ())
    if selected == ("cleaned",):
        return "cleaned only"
    if selected == ("phrases", "cleaned"):
        return "phrases + cleaned"
    return "phrases only"


def setup_name(settings):
    pathway = settings.get("pathway", "quantum")
    if pathway == "none":
        return "No encoding/decoding"
    if pathway == "frozen-encoding-decoding":
        return "Frozen encoder/decoder"
    ansatz = settings.get("ansatz", "trainable")
    return {"trainable": "Full trainable", "zero": "No circuit", "frozen": "Frozen circuit"}[ansatz]


def baseline(run):
    saved = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    targets = saved["targets"]
    train_ids = torch.tensor(saved["train_ids"])
    test_ids = torch.tensor(saved["test_ids"])
    counts = torch.bincount(targets[train_ids])
    ranking = counts.argsort(descending=True)
    held = targets[test_ids]
    return {
        "setup": "Baseline",
        "dataset": dataset_name(saved["settings"]),
        "best_epoch": 0,
        "best_validation": {},
        "test": {
            "cosine_top1": float((held == ranking[0]).float().mean()),
            "cosine_top5": float((held[:, None] == ranking[:5]).any(1).float().mean()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/semantic-matrix-report"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    first_by_dataset = {}
    for root in args.roots:
        for run in sorted(root.glob("**/semantic-causal-*")):
            summary_path = run / "summary.json"
            history_path = run / "history.json"
            test_path = run / "test.json"
            if not (summary_path.exists() and history_path.exists() and test_path.exists()):
                continue
            summary = json.loads(summary_path.read_text())
            settings = summary["settings"]
            dataset = dataset_name(settings)
            first_by_dataset.setdefault(dataset, run)
            history = json.loads(history_path.read_text())
            best = min(history, key=lambda row: row["validation"]["cross_entropy"])
            test = json.loads(test_path.read_text())
            rows.append(
                {
                    "dataset": dataset,
                    "setup": setup_name(settings),
                    "best_epoch": best["epoch"],
                    "best_validation": best["validation"],
                    "test": {
                        "cosine_top1": test["cosine_top1"],
                        "cosine_top5": test["cosine_top5"],
                    },
                    "run": str(run),
                }
            )
    rows.extend(baseline(run) for run in first_by_dataset.values())
    rows.sort(key=lambda row: (row["dataset"], row["setup"]))
    (args.output / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n")

    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["dataset"]][row["setup"]] = row
    datasets = sorted(grouped)
    setups = [
        "Baseline",
        "No encoding/decoding",
        "Frozen encoder/decoder",
        "No circuit",
        "Frozen circuit",
        "Full trainable",
    ]
    x = np.arange(len(datasets))
    width = 0.12
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for index, setup in enumerate(setups):
        top1 = [
            grouped[dataset].get(setup, {}).get("test", {}).get("cosine_top1", np.nan)
            for dataset in datasets
        ]
        top5 = [
            grouped[dataset].get(setup, {}).get("test", {}).get("cosine_top5", np.nan)
            for dataset in datasets
        ]
        axes[0].bar(x + (index - 2.5) * width, top1, width, label=setup)
        axes[1].bar(x + (index - 2.5) * width, top5, width, label=setup)
    axes[0].set_ylabel("test cosine top-1")
    axes[1].set_ylabel("test cosine top-5")
    for axis in axes:
        axis.set_xticks(x, datasets, rotation=15)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.savefig(args.output / "test-cosine-bars.png", dpi=160)

    figure, axes = plt.subplots(
        1, len(datasets), figsize=(16, 4), sharey=True, constrained_layout=True
    )
    if len(datasets) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, datasets, strict=True):
        for setup in setups:
            row = grouped[dataset].get(setup)
            if row and row["best_validation"]:
                axis.scatter(row["best_epoch"], row["best_validation"]["top1"], label=setup)
        axis.set_title(dataset)
        axis.set_xlabel("best epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("best validation cosine top-1")
    axes[-1].legend(fontsize=8)
    figure.savefig(args.output / "best-validation-cosine.png", dpi=160)
    print(f"Saved reports under {args.output}")


if __name__ == "__main__":
    main()
