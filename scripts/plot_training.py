"""Plot QCSE training metrics from the line-oriented training log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_PATTERN = re.compile(
    r"^Epoch\s+(?P<epoch>\d+):\s+"
    r"train BCE=(?P<train_bce>[0-9.eE+-]+),\s+"
    r"test BCE=(?P<test_bce>[0-9.eE+-]+),\s+"
    r"exact word=(?P<exact_word>[0-9.eE+-]+),\s+"
    r"paper score=(?P<paper_score>[0-9.eE+-]+)\s*$"
)


def read_log(path: Path) -> list[dict[str, float]]:
    """Read metric rows from *path*, ignoring non-metric lines."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LOG_PATTERN.match(line)
        if match is None:
            continue
        row = {key: float(value) for key, value in match.groupdict().items()}
        rows.append(row)
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
    for axis, (title, keys, labels, y_label) in zip(axes, panels):
        for key, label in zip(keys, labels):
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
    parser.add_argument("log", type=Path, help="Training log, e.g. train.txt")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path (default: <log-stem>_metrics.png beside the log)",
    )
    args = parser.parse_args()
    output = args.output or args.log.with_name(f"{args.log.stem}_metrics.png")
    plot_training(read_log(args.log), output)
    print(f"Saved training plot to {output}")


if __name__ == "__main__":
    main()
