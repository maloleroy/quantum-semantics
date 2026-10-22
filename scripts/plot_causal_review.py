"""Plot both completed mechanism tracks and the measured microbatch opportunity."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, nargs="?", default=Path("results/robust-causal-controls")
    )
    root = parser.parse_args().root
    result = json.loads((root / "audit.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    controls = ["depth2-init0.05", "local", "frozen", "xx", "zz"]
    names = ["CRZ", "Local", "Frozen CRZ", "XX", "ZZ"]
    for recipe, offset, color, label in [
        ("L4", -0.1, "#31688e", "Causal L4"),
        ("L2R2", 0.1, "#d55e00", "Bidirectional L2R2"),
    ]:
        for ax, mi, multiplier in [(axes[0, 0], 2, 100), (axes[0, 1], 0, 1)]:
            for index, control in enumerate(controls):
                c = result["contrasts"][f"{recipe}:hybrid-q8-{control}-minus-identity"]
                lo, hi = [multiplier * c["seed_sentence_nominal95"][k][mi] for k in ("low", "high")]
                ax.plot(
                    [lo, hi], [index + offset] * 2, color=color, label=label if index == 0 else None
                )
                ax.scatter(multiplier * c["mean"][mi], index + offset, color=color)
                ax.scatter(
                    [multiplier * v[mi] for v in c["per_seed"]],
                    [index + offset] * 2,
                    marker="x",
                    color=color,
                    s=25,
                )
            ax.set(yticks=range(5), yticklabels=names)
            ax.axvline(0, color="gray", linestyle="--")
            ax.set_ylim(4.5, -0.5)
    axes[0, 0].set(title="Circuit − identity: top-5", xlabel="Percentage points; larger is better")
    axes[0, 1].set(title="Circuit − identity: NLL", xlabel="NLL difference; smaller is better")
    axes[0, 0].legend(fontsize=9)
    main = ["mlp-32", "bert-32", "hybrid-q8-depth2-init0.05", "hybrid-q8-identity"]
    for recipe, offset, color, label in [
        ("L4", -0.17, "#31688e", "Causal"),
        ("L2R2", 0.17, "#d55e00", "Bidirectional"),
    ]:
        rows = {
            r["arm"]: r
            for r in result["pilot"]["speedups"]
            if r["recipe"] == recipe and r["batch"] == 64
        }
        axes[1, 0].barh(
            np.arange(4) + offset,
            [rows[a]["speedup_8_to_64"] for a in main],
            height=0.3,
            color=color,
            label=label,
        )
    axes[1, 0].set(
        yticks=range(4),
        yticklabels=["MLP-32", "BERT-32", "q8 CRZ", "q8 identity"],
        xlabel="Training-step speedup (microbatch8 / microbatch64)",
        title="One-MIG pilot, batch64: whole fits not measured",
        xlim=(0, 9),
    )
    axes[1, 0].invert_yaxis()
    axes[1, 0].legend(fontsize=9)
    for index, recipe in enumerate(("L4", "L2R2")):
        values = []
        for arm in main:
            selected = [r for r in result["rows"] if (r["recipe"], r["arm"]) == (recipe, arm)]
            values.append(100 * np.mean([r["top5"] for r in selected]))
        axes[1, 1].barh(
            np.arange(4) + (index - 0.5) * 0.34,
            values,
            height=0.3,
            color=["#31688e", "#d55e00"][index],
            label=recipe,
        )
    axes[1, 1].set(
        yticks=range(4),
        yticklabels=["MLP-32", "BERT-32*", "q8 CRZ", "q8 identity"],
        title="Classical references still lead both tracks",
        xlabel="Validation top-5 (%)",
        xlim=(0, 85),
    )
    axes[1, 1].invert_yaxis()
    axes[1, 1].legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle(
        "Two fitted seeds, one reused validation split: exploratory mechanism evidence\n"
        "Nominal 95% paired seed/sentence intervals; crosses show individual seed effects",
        fontsize=12,
    )
    fig.supxlabel(
        "* Causal-track BERT sees left context only but retains full attention; it is not GPT.",
        fontsize=10,
    )
    fig.savefig(root / "mechanisms-and-compute.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
