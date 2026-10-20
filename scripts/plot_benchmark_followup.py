"""Render audited follow-up outcomes; intervals are the exported conditional intervals."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, default=Path("results/robust-followup"), nargs="?")
    root = parser.parse_args().root
    summary = json.loads((root / "summary.json").read_text())
    names = [
        "mlp-32",
        "bert-32",
        "hybrid-q8-depth2-init0.05",
        "hybrid-q8-identity",
        "qcse-context-angular",
    ]
    labels = ["MLP-32", "BERT-32", "q8 CRZ", "q8 identity", "QCSE angular"]
    colors = {"causal": "#276397", "cbow": "#d37529"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for ax, objective in zip(axes[0], ("causal", "cbow"), strict=True):
        study = summary["studies"]["confirm-" + objective]
        records = {r["arm"]: r for r in study["heldout"]}
        means = [100 * records[n]["ledger_mean"]["top1"] for n in names]
        ax.barh(labels, means, color=colors[objective], alpha=0.8)
        for i, n in enumerate(names):
            values = [100 * v["top1"] for v in records[n]["seed_metrics"]]
            ax.scatter(values, i + np.array([-0.1, 0, 0.1]), s=17, color="black", zorder=3)
            ax.text(means[i] + 0.6, i, f"{means[i]:.2f}%", va="center", fontsize=9)
        ax.axvline(
            100 * study["test_frequency"]["top1"], color="gray", linestyle=":", label="Frequency"
        )
        ax.set(title=f"{objective.upper()} held-out top-1", xlabel="Accuracy (%)", xlim=(0, 63))
        ax.invert_yaxis()
        ax.legend(loc="lower right", fontsize=8)
    ax = axes[1, 0]
    for i, objective in enumerate(("causal", "cbow")):
        study = summary["studies"]["confirm-" + objective]
        p = next(r for r in study["reported_intervals"] if r["primary"])
        center, low, high = [100 * p[k][1] for k in ("difference", "low", "high")]
        ax.errorbar(
            center,
            i,
            xerr=[[center - low], [high - center]],
            fmt="o",
            capsize=5,
            color=colors[objective],
        )
        pair = next(r for r in study["pairs"] if not r["cv"] and r["left"] == names[2])
        ax.scatter(pair["seed_mean_top1_pp"], i + np.array([-0.1, 0, 0.1]), color="black", s=15)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set(
        yticks=[0, 1],
        yticklabels=["Causal", "CBOW"],
        ylim=(-0.5, 1.5),
        title="CRZ − identity: primary comparison",
        xlabel="Top-1 difference (percentage points)",
    )
    ax.text(
        0.02,
        0.03,
        "97.5% seed/sentence intervals; black dots = training seeds",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax = axes[1, 1]
    for objective, shift in (("causal", -0.08), ("cbow", 0.08)):
        study = summary["studies"]["confirm-" + objective]
        by_name = {a["arm"]["name"]: a for a in study["arms"]}
        costs = [by_name[n]["median_cv_seconds"] / 60 for n in names]
        ax.scatter(
            costs, np.arange(len(names)) + shift, label=objective.upper(), color=colors[objective]
        )
    ax.set(
        yticks=np.arange(5),
        yticklabels=labels,
        xscale="log",
        xlabel="Measured median CV allocation minutes (log scale)",
        title="Cost of the confirmed architectures",
    )
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    fig.suptitle("Completed follow-up: paired controls and matched-budget classical baselines")
    fig.savefig(root / "heldout-and-cost.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, (depth, scale) in zip(axes.flat, ((2, 0.05), (2, 1), (8, 0.05), (8, 1)), strict=True):
        for objective in ("causal", "cbow"):
            for stage, style in (("refine", "--"), ("confirm", "-")):
                rows = summary["studies"][f"{stage}-{objective}"]["nlp_probe"]
                rows = sorted(
                    [r for r in rows if r["layers"] == depth and r["init_scale"] == scale],
                    key=lambda r: r["qubits"],
                )
                ax.plot(
                    [r["qubits"] for r in rows],
                    [r["first_rx_gradient_variance"] for r in rows],
                    style,
                    marker="o",
                    color=colors[objective],
                    label=f"{objective} {stage}",
                )
        ax.set(
            yscale="log",
            xticks=[2, 4, 8, 12],
            xlabel="Qubits",
            ylabel="First-RX gradient variance",
            title=f"Depth {depth}, initialization std {scale:g}",
        )
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Actual train-fold CE probes: 32 circuit initializations per condition\n"
        "One batch/head per condition; changing qubit count also changes encoder/head shape",
        fontsize=12,
    )
    fig.savefig(root / "nlp-gradient-probes.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
