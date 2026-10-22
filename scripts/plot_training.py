"""Plot QCSE training metrics from a native ``run.npz`` archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_run(path: Path) -> list[dict]:
    """Read the history embedded in a native training run archive."""
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        rows = json.loads(str(archive["history"]))
    if not rows:
        raise ValueError(f"No training metric rows found in {path}")
    epochs = [row["epoch"] for row in rows]
    if epochs != sorted(set(epochs)):
        raise ValueError(f"Epoch values must be strictly increasing in {path}")
    return rows


def plot_training(rows: list[dict], output: Path) -> None:
    """Create and save a three-panel training-history figure."""
    epochs = [row["epoch"] for row in rows]
    evaluation = next((name for name in ("validation", "test") if name in rows[0]), "train")
    splits = ("train", evaluation) if evaluation != "train" else ("train",)
    panels = (
        ("BCE loss", "bce", splits, "Loss"),
        ("Exact-word accuracy", "exact_word_accuracy", (evaluation,), "Accuracy"),
        ("Paper similarity score", "paper_similarity_accuracy", (evaluation,), "Score"),
    )
    colors = {
        "train": "#2563eb",
        "validation": "#059669",
        "test": "#dc2626",
    }

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    scope = " (fixed monitoring samples)" if "metric_examples" in rows[0] else ""
    figure.suptitle(f"QCSE training history{scope}", fontsize=16, fontweight="bold")
    for axis, (title, metric, names, y_label) in zip(axes, panels, strict=True):
        for name in names:
            values = [row[name][metric] for row in rows]
            axis.plot(epochs, values, color=colors[name], linewidth=2, label=name.title())
            axis.scatter(epochs[-1], values[-1], color=colors[name], s=28, zorder=3)
            axis.annotate(
                f"{values[-1]:.6f}",
                xy=(epochs[-1], values[-1]),
                xytext=(-8, 8),
                textcoords="offset points",
                color=colors[name],
                ha="right",
                fontsize=9,
            )
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, ncol=len(names), loc="best")
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
