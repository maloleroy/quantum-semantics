"""Plot paired uncertainty and target-population sensitivity from the audited export."""

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
        "root", type=Path, nargs="?", default=Path("results/robust-followup-prediction-review")
    )
    root = parser.parse_args().root
    summary = json.loads((root / "summary.json").read_text())
    populations = ["causal/all", "cbow/all", "cbow/common"]
    labels = ["Causal: full task", "CBOW: full task", "CBOW: positions > 0"]
    contrasts = {
        r["population"]: r
        for r in summary["contrasts"]
        if r["left"] == "hybrid-q8-depth2-init0.05" and r["right"] == "hybrid-q8-identity"
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    for ax, metric, title in zip(
        axes,
        [1, 2],
        ["Top-1: primary on the original full tasks", "Top-5: secondary / exploratory"],
        strict=True,
    ):
        for i, population in enumerate(populations):
            c = contrasts[population]
            center = 100 * c["difference"][metric]
            for key, offset, color, label in [
                ("paired", -0.08, "#276397", "Paired seeds + sentences"),
                ("sentences_only", 0.08, "#d37529", "Sentences only; fitted seeds fixed"),
            ]:
                lo, hi = [100 * c[key][k][metric] for k in ["low", "high"]]
                ax.errorbar(
                    center,
                    i + offset,
                    xerr=[[center - lo], [hi - center]],
                    fmt="o",
                    color=color,
                    capsize=4,
                    label=label if i == 0 else None,
                )
        ax.axvline(0, color="gray", linestyle="--")
        ax.set(
            yticks=[0, 1, 2],
            yticklabels=labels,
            title=title,
            xlabel="CRZ − identity (percentage points)",
            ylim=(-0.5, 2.5),
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Nominal 97.5% intervals: 10,000 draws, three trained seeds, fixed split\n"
        "Common-position comparisons are post hoc; sentence-only intervals "
        "omit training-seed variation",
        fontsize=11,
    )
    fig.savefig(root / "paired-uncertainty.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 6), layout="constrained", gridspec_kw={"width_ratios": [1.4, 1]}
    )
    names = [
        "mlp-32",
        "bert-32",
        "hybrid-q8-depth2-init0.05",
        "hybrid-q8-identity",
        "qcse-context-angular",
    ]
    display = ["MLP-32", "BERT-32", "q8 CRZ", "q8 identity", "QCSE angular"]
    rows = {(r["population"], r["arm"]): r for r in summary["scores"]}
    for population, offset, color, label in [
        ("causal/all", -0.23, "#276397", "Causal"),
        ("cbow/all", 0, "#d37529", "CBOW all targets"),
        ("cbow/common", 0.23, "#28856b", "CBOW matching positions"),
    ]:
        values = [100 * rows[population, n]["mean"][2] for n in names]
        axes[0].barh(np.arange(5) + offset, values, height=0.21, color=color, label=label)
    axes[0].set(
        yticks=np.arange(5),
        yticklabels=display,
        xlabel="Held-out top-5 (%)",
        title="The same sentence split, different target populations",
        xlim=(0, 85),
    )
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, loc="lower right")
    words = summary["checks"]["cbow"]["first_words"][:6]
    axes[1].barh([w for w, _ in words], [100 * n / 40434 for _, n in words], color="#718095")
    axes[1].invert_yaxis()
    axes[1].set(
        xlabel="Share of sentence-initial targets (%)", title="Sentence-initial corpus composition"
    )
    axes[1].text(
        0.98,
        0.08,
        "“i” + “tom”: 70.7%\nDescriptive corpus statistic",
        ha="right",
        transform=axes[1].transAxes,
        fontsize=10,
    )
    fig.suptitle("Matching positions removes most of QCSE’s apparent CBOW top-5 increase")
    fig.savefig(root / "target-populations.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
