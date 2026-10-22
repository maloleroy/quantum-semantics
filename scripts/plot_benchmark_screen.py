"""Render the audited screen's descriptive accuracy, cost and gradient figures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(root):
    summary = json.loads((root / "summary.json").read_text())
    arms = {v["arm"]["name"]: v for v in summary["arms"]}
    names = [
        "bert-32",
        "hybrid-q8-identity",
        "hybrid-q8-depth2-init0.05",
        "mlp",
        "bert-16-depth-4",
        "hybrid-depth-4",
        "hybrid-crz-chain",
        "hybrid-identity",
        "bert-16",
        "qcse-context-angular",
        "qcse-crz-chain",
        "qcse-depth-8",
        "qcse-permuted-ids",
        "mean",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), constrained_layout=True)
    colors = ["#536f8c" if arms[n]["arm"]["model"] != "qcse" else "#a17350" for n in names]
    axes[0].barh(
        names,
        [100 * arms[n]["mean"]["top1"] for n in names],
        xerr=[100 * arms[n]["sd"]["top1"] for n in names],
        color=colors,
        capsize=3,
    )
    axes[0].axvline(
        100 * summary["frequency"]["top1"],
        color="#777777",
        linestyle="--",
        label="Train-frequency baseline",
    )
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_xlabel("Full-fold validation top-1 (%) · bars: SD across 3 folds")
    axes[1].barh(names, [arms[n]["median_seconds"] / 3600 for n in names], color=colors)
    axes[1].set_xlabel("Measured median allocation cost per fit (hours)")
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("DGX A100 1g.10gb · 100 × 20,000 draws · seed 42 · selected arms")
    fig.savefig(root / "accuracy-and-cost.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name in [
        "bert-32",
        "mlp",
        "hybrid-q8-identity",
        "hybrid-q8-depth2-init0.05",
        "qcse-context-angular",
    ]:
        curves = arms[name]["curves"]
        axes[0].plot(
            [v["epoch"] for v in curves], [100 * v["top1"] for v in curves], marker="o", label=name
        )
    axes[0].set(
        xlabel="Sampled epoch",
        ylabel="Validation-monitor top-1 (%)",
        title="Fixed 2,048-example monitors; descriptive curves",
    )
    axes[0].legend(fontsize=7)
    probe = json.loads((root / "synthetic-probe.json").read_text())
    for cost in ("local", "global"):
        rows = [
            v
            for v in probe["rows"]
            if v["cost"] == cost and v["layers"] == 8 and v["init_scale"] == 1.0
        ]
        axes[1].semilogy(
            [v["qubits"] for v in rows],
            [v["first_rx_gradient_variance"] for v in rows],
            marker="o",
            label=cost,
        )
    axes[1].set(
        xlabel="Qubits",
        ylabel="Variance of first RX gradient across 32 seeds",
        title="Synthetic probe · depth 8 · initialization std 1",
    )
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(root / "learning-and-trainability.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("report", type=Path)
    plot(cli.parse_args().report)
