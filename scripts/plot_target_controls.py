"""Plot the independently audited development scores and seed sensitivity."""

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
        "root", type=Path, nargs="?", default=Path("results/robust-target-controls")
    )
    root = parser.parse_args().root
    result = json.loads((root / "audit.json").read_text())
    arms = ["mlp-32", "bert-32", "hybrid-q8-depth2-init0.05", "hybrid-q8-identity"]
    names = ["MLP-32", "BERT-32*", "q8 CRZ", "q8 identity"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, recipe, title in zip(
        axes[:2],
        ["L4", "L2R2"],
        ["Left-only: four preceding words", "Bidirectional: two left + two right"],
        strict=True,
    ):
        for index, arm in enumerate(arms):
            rows = [r for r in result["rows"] if r["recipe"] == recipe and r["arm"] == arm]
            for metric, offset, color in [("top1", -0.16, "#31688e"), ("top5", 0.16, "#35b779")]:
                values = [100 * r[metric] for r in rows]
                ax.barh(
                    index + offset,
                    np.mean(values),
                    height=0.28,
                    color=color,
                    label=metric if index == 0 else None,
                )
                ax.scatter(values, [index + offset] * 2, s=13, color="black", zorder=3)
        ax.set(
            yticks=range(4),
            yticklabels=names,
            xlim=(0, 85),
            title=title,
            xlabel="Validation accuracy (%); dots = seeds 47,48",
        )
        ax.invert_yaxis()
        ax.legend(fontsize=8)
    controls = ["depth2-init0.05", "local", "frozen", "xx", "zz"]
    for index, name in enumerate(controls):
        contrast = result["contrasts"][f"L2R2:hybrid-q8-{name}-minus-identity"]
        lo, hi = [100 * contrast["seed_sentence_nominal95"][k][2] for k in ("low", "high")]
        center = 100 * contrast["mean"][2]
        axes[2].plot([lo, hi], [index, index], color="#31688e")
        axes[2].scatter([center], [index], color="#31688e", zorder=3)
        axes[2].scatter(
            [100 * v[2] for v in contrast["per_seed"]],
            [index] * 2,
            marker="x",
            color="#d55e00",
            zorder=4,
        )
    axes[2].axvline(0, color="gray", linestyle="--")
    axes[2].set(
        yticks=range(5),
        yticklabels=["CRZ", "Local", "Frozen CRZ", "XX", "ZZ"],
        title="Bidirectional circuit − identity",
        xlabel="Top-5 difference (percentage points)",
    )
    axes[2].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle(
        "Development evidence: two seeds, one reused validation split\n"
        "Intervals: nominal 95% paired seed/sentence bootstrap; crosses: individual seeds",
        fontsize=12,
    )
    fig.text(
        0.01,
        -0.035,
        "*BERT masked-token architecture receives only the specified words; "
        "the left-only version is not a GPT decoder. No new test scores.",
        fontsize=9,
    )
    fig.savefig(root / "dual-track-controls.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
