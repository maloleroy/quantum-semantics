"""Create annotated heatmaps from a CSV parameter grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = (
    ("test_bce", "Test BCE", "BCE", ".3f", "viridis"),
    ("bit_accuracy", "Bit accuracy", "Accuracy", ".1%", "magma"),
    ("exact_word_accuracy", "Exact-word accuracy", "Accuracy", ".2%", "plasma"),
    (
        "paper_similarity_accuracy",
        "Paper similarity accuracy",
        "Accuracy",
        ".1%",
        "cividis",
    ),
)


def read_grid(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read and validate a rectangular CSV parameter grid."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    required = {"alpha", "window", *(metric[0] for metric in METRICS)}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")

    alphas = np.array(sorted({float(row["alpha"]) for row in rows}))
    windows = np.array(sorted({int(row["window"]) for row in rows}))
    expected_rows = len(alphas) * len(windows)
    if len(rows) != expected_rows:
        raise ValueError(
            f"Expected a complete {len(alphas)}x{len(windows)} grid in {path}, "
            f"found {len(rows)} rows"
        )

    values = {metric[0]: np.full((len(alphas), len(windows)), np.nan) for metric in METRICS}
    locations: set[tuple[float, int]] = set()
    alpha_index = {value: index for index, value in enumerate(alphas)}
    window_index = {value: index for index, value in enumerate(windows)}
    for row in rows:
        alpha = float(row["alpha"])
        window = int(row["window"])
        location = (alpha, window)
        if location in locations:
            raise ValueError(f"Duplicate grid point in {path}: alpha={alpha}, window={window}")
        locations.add(location)
        for key, _, _, _, _ in METRICS:
            values[key][alpha_index[alpha], window_index[window]] = float(row[key])

    if any(np.isnan(matrix).any() for matrix in values.values()):
        raise ValueError(f"Grid contains missing values in {path}")
    return alphas, windows, values


def format_value(value: float, format_spec: str) -> str:
    """Format a cell value while keeping percentages legible."""
    return format(value, format_spec)


def plot_grid(path: Path, output: Path) -> None:
    """Create and save one annotated heatmap per metric."""
    alphas, windows, values = read_grid(path)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    figure.suptitle("QCSE parameter grid", fontsize=16, fontweight="bold")

    for axis, (key, title, colorbar_label, format_spec, cmap) in zip(
        axes.flat, METRICS, strict=True
    ):
        matrix = values[key]
        image = axis.imshow(matrix, cmap=cmap, aspect="auto")
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_xlabel("Window")
        axis.set_ylabel("Alpha")
        axis.set_xticks(range(len(windows)), labels=[str(value) for value in windows])
        axis.set_yticks(range(len(alphas)), labels=[f"{value:g}" for value in alphas])
        axis.tick_params(length=0)

        midpoint = (matrix.min() + matrix.max()) / 2
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                color = "white" if matrix[row_index, column_index] < midpoint else "black"
                axis.text(
                    column_index,
                    row_index,
                    format_value(matrix[row_index, column_index], format_spec),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )

        colorbar = figure.colorbar(image, ax=axis, pad=0.03, fraction=0.046)
        colorbar.set_label(colorbar_label)
        colorbar.ax.tick_params(length=0)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=Path, help="CSV grid file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path (default: outputs/<grid-stem>_heatmap.png)",
    )
    args = parser.parse_args()
    output = args.output or Path("outputs") / f"{args.grid.stem}_heatmap.png"
    plot_grid(args.grid, output)
    print(f"Saved grid visualization to {output}")


if __name__ == "__main__":
    main()
