"""Plot histories from semantic decoder ablation runs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/semantic-report"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    curves = []
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        settings = summary["settings"]
        history = json.loads((run / "history.json").read_text())
        label = f"{settings['ansatz']}/{settings['trainable']}"
        curves.append((run, label, history, summary["parameters"]))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for _, label, history, _ in curves:
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["validation"]["cross_entropy"] for row in history], label=label)
        axes[1].plot(epochs, [row["validation"]["top1"] for row in history], label=label)
        axes[2].plot(epochs, [row["validation"]["cosine_top1"] for row in history], label=label)
    axes[0].set_ylabel("validation cross-entropy")
    axes[1].set_ylabel("dot-product top-1")
    axes[2].set_ylabel("cosine top-1")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.savefig(args.output / "validation-ablation.png", dpi=160)

    rows = []
    for _, label, history, parameters in curves:
        best = min(history, key=lambda row: row["validation"]["cross_entropy"])
        rows.append(
            {
                "label": label,
                "best_epoch": best["epoch"],
                "best_validation": best["validation"],
                "final_validation": history[-1]["validation"],
                "parameters": parameters,
            }
        )
    (args.output / "ablation-summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Saved {args.output / 'validation-ablation.png'}")


if __name__ == "__main__":
    main()
