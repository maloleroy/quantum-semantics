"""Plot QCSE training metrics from a native ``run.npz`` archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_run(path: Path) -> list[dict[str, float]]:
    """Read the history embedded in a native training run archive."""
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        rows = json.loads(str(archive["history"]))
    rows = [
        {
            "epoch": row["epoch"],
            "train_bce": row["train"]["bce"],
            "test_bce": row["test"]["bce"],
            "exact_word": row["test"]["exact_word_accuracy"],
            "paper_score": row["test"]["paper_similarity_accuracy"],
        }
        for row in rows
    ]
    if not rows:
        raise ValueError(f"No training metric rows found in {path}")
    epochs = [row["epoch"] for row in rows]
    if epochs != sorted(set(epochs)):
        raise ValueError(f"Epoch values must be strictly increasing in {path}")
    return rows


def plot_training(rows: list[dict[str, float]], output: Path) -> None:
    """Create and save a three-panel training-history figure."""
    epochs = [row["epoch"] for row in rows]
    panels = (
        ("BCE loss", ("train_bce", "test_bce"), ("Train", "Test"), "Loss"),
        ("Exact-word accuracy", ("exact_word",), ("Test",), "Accuracy"),
        ("Paper similarity score", ("paper_score",), ("Test",), "Score"),
    )
    colors = {
        "train_bce": "#2563eb",
        "test_bce": "#dc2626",
        "exact_word": "#059669",
        "paper_score": "#9333ea",
    }

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    figure.suptitle("QCSE training history", fontsize=16, fontweight="bold")
    for axis, (title, keys, labels, y_label) in zip(axes, panels, strict=True):
        for key, label in zip(keys, labels, strict=True):
            values = [row[key] for row in rows]
            axis.plot(epochs, values, color=colors[key], linewidth=2, label=label)
            axis.scatter(epochs[-1], values[-1], color=colors[key], s=28, zorder=3)
            axis.annotate(
                f"{values[-1]:.3f}",
                xy=(epochs[-1], values[-1]),
                xytext=(-8, 8),
                textcoords="offset points",
                color=colors[key],
                ha="right",
                fontsize=9,
            )
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, ncol=len(keys), loc="best")
        axis.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Epoch")
    axes[-1].set_xlim(epochs[0], epochs[-1])
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Training run archive, e.g. outputs/train/run.npz")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path (default: <log-stem>_metrics.png beside the log)",
    )
    args = parser.parse_args()
    output = args.output or args.log.with_name(f"{args.log.stem}_metrics.png")
    plot_training(read_run(args.log), output)
    print(f"Saved training plot to {output}")


if __name__ == "__main__":
    main()
