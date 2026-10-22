"""Aggregate the 30-epoch full-circuit versus no-circuit sweep."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/semantic-hyper-report"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for summary_path in sorted(args.root.glob("**/summary.json")):
        run = summary_path.parent
        history_path, test_path = run / "history.json", run / "test.json"
        if not (history_path.exists() and test_path.exists()):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        settings = summary["settings"]
        history = json.loads(history_path.read_text(encoding="utf-8"))
        test = json.loads(test_path.read_text(encoding="utf-8"))
        best = min(history, key=lambda row: row["validation"]["cross_entropy"])
        rows.append(
            {
                "config": {
                    "alpha": settings["alpha"],
                    "learning_rate": settings["learning_rate"],
                    "layers": settings["layers"],
                    "batch_size": settings["batch_size"],
                },
                "ansatz": "full trainable" if settings["ansatz"] == "trainable" else "no circuit",
                "best_epoch": best["epoch"],
                "best_validation": {
                    "cosine_top1": best["validation"]["cosine_top1"],
                    "cosine_top5": best["validation"]["cosine_top5"],
                },
                "test": {
                    "cosine_top1": test["cosine_top1"],
                    "cosine_top5": test["cosine_top5"],
                },
                "run": str(run),
            }
        )
    if not rows:
        raise SystemExit(f"No complete runs found under {args.root}")

    rows.sort(
        key=lambda row: (
            row["config"]["alpha"],
            row["config"]["learning_rate"],
            row["config"]["layers"],
            row["config"]["batch_size"],
            row["ansatz"],
        )
    )
    (args.output / "hyper-grid.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )

    grouped = defaultdict(dict)
    for row in rows:
        key = tuple(row["config"].values())
        grouped[key][row["ansatz"]] = row
    labels, deltas, full_top1, zero_top1, full_top5, zero_top5 = [], [], [], [], [], []
    for config, pair in grouped.items():
        if len(pair) != 2:
            continue
        alpha, learning_rate, layers, batch_size = config
        label = f"a={alpha:g}\nlr={learning_rate:g}\nL={layers}, B={batch_size}"
        labels.append(label)
        full, zero = pair["full trainable"], pair["no circuit"]
        full_top1.append(full["test"]["cosine_top1"])
        zero_top1.append(zero["test"]["cosine_top1"])
        full_top5.append(full["test"]["cosine_top5"])
        zero_top5.append(zero["test"]["cosine_top5"])
        deltas.append(zero["test"]["cosine_top1"] - full["test"]["cosine_top1"])

    x = np.arange(len(labels))
    width = 0.38
    figure, axes = plt.subplots(
        2, 1, figsize=(max(12, len(labels) * 0.65), 9), constrained_layout=True
    )
    axes[0].bar(x - width / 2, full_top1, width, label="full trainable")
    axes[0].bar(x + width / 2, zero_top1, width, label="no circuit")
    axes[0].set_ylabel("test cosine top-1")
    axes[0].set_title("30-epoch hyperparameter sweep")
    axes[1].bar(x - width / 2, full_top5, width, label="full trainable")
    axes[1].bar(x + width / 2, zero_top5, width, label="no circuit")
    axes[1].set_ylabel("test cosine top-5")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=0, fontsize=8)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.savefig(args.output / "test-cosine-hyper-bars.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(max(10, len(labels) * 0.6), 4), constrained_layout=True)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.bar(x, deltas, color=np.where(np.asarray(deltas) >= 0, "#4472c4", "#c0504d"))
    axis.set_xticks(x, labels, fontsize=8)
    axis.set_ylabel("no circuit minus full trainable\ntest cosine top-1")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(args.output / "no-circuit-delta.png", dpi=160)
    plt.close(figure)
    print(f"Saved reports under {args.output}")


if __name__ == "__main__":
    main()
