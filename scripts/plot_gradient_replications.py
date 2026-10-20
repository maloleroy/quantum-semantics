"""Descriptive crossed-condition ranges, not confidence intervals."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("results/gradient-replicated-10324")
    rows = json.loads((root / "summary.json").read_text())["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    qubits = [2, 4, 8, 12]
    for axrow, objective in zip(axes, ("causal", "cbow"), strict=True):
        for ax, key, label in zip(
            axrow,
            ("mean_gradient_variance", "rms_gradient"),
            ("Variance across circuit initializations", "RMS of actual circuit gradients"),
            strict=True,
        ):
            for depth, scale, color, style in [
                (2, 0.05, "#2166ac", "-"),
                (8, 0.05, "#67a9cf", "--"),
                (2, 1.0, "#b2182b", "-"),
                (8, 1.0, "#ef8a62", "--"),
            ]:
                values = np.array(
                    [
                        [
                            r[key]
                            for r in rows
                            if r["objective"] == objective
                            and (r["qubits"], r["layers"], r["init_scale"]) == (q, depth, scale)
                        ]
                        for q in qubits
                    ]
                )
                ax.plot(
                    qubits,
                    np.median(values, axis=1),
                    marker="o",
                    color=color,
                    linestyle=style,
                    label=f"depth {depth}, init std {scale:g}",
                )
                ax.fill_between(qubits, values.min(1), values.max(1), color=color, alpha=0.08)
            ax.set(
                yscale="log", xticks=qubits, xlabel="Qubits", ylabel=label, title=objective.upper()
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Actual NLP loss: medians and full ranges across 3 batches × 3 heads\n"
        "32 paired circuit seeds per condition; ranges are not confidence intervals. "
        "Encoder/head shape changes with q.",
        fontsize=11,
    )
    fig.savefig(root / "initial-gradients.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
