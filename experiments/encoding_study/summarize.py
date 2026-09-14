"""Aggregate recorded runs and paired sentence-bootstrap comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qcse.data import load_phrases

from .encoders import Features
from .run import prepare, write_json

ROOT = Path("experiments/encoding_study")


def aggregate(stage):
    grouped = {}
    for path in (ROOT / "results" / stage).glob("*_seed*.json"):
        row = json.loads(path.read_text())
        grouped.setdefault(row["method"], []).append(row)
    summary = {}
    for method, rows in grouped.items():
        summary[method] = {
            "seeds": sorted(row["seed"] for row in rows),
            "parameters": rows[0]["parameters"],
            "selected_epochs": [r["selected_epoch"] for r in sorted(rows, key=lambda x: x["seed"])],
            "seconds_total": sum(r["seconds"] for r in rows),
        }
        for partition in rows[0]["metrics"]:
            summary[method][partition] = {
                metric: {
                    "mean": float(np.mean([r["metrics"][partition][metric] for r in rows])),
                    "sd": float(np.std([r["metrics"][partition][metric] for r in rows], ddof=1)),
                }
                for metric in rows[0]["metrics"][partition]
            }
    return summary


def paired_intervals(confirmation, reference="exponential"):
    sentences = load_phrases()
    sentence_keys = {tuple(s): i for i, s in enumerate(sentences)}
    reference_p = None
    if reference in ("bit_prior", "bigram_bits"):
        cfg = json.loads((ROOT / "results/confirmation/config.json").read_text())
        vocabulary, examples, train, _, test = prepare(argparse.Namespace(**cfg))
        features = Features(examples, train, len(vocabulary), 10)
        reference_p = (
            np.broadcast_to(features.bit_prior, (len(test), 10))
            if reference == "bit_prior"
            else features.bigram_bits[[examples[i].context[-1] for i in test]]
        )
    result = {}
    for method, row in confirmation.items():
        bce, accuracy = [], []
        for seed in row["seeds"]:
            with np.load(ROOT / "results/confirmation" / f"{method}_seed{seed}.npz") as a:
                p = np.clip(a["test_probabilities"], 1e-10, 1 - 1e-10)
                target = a["test_targets"]
                ids = a["test_ids"]
                groups = np.array(
                    [sentence_keys[tuple(sentences[i])] for i in a["test_sentence_ids"]]
                )
            if reference_p is None:
                with np.load(ROOT / "results/confirmation" / f"{reference}_seed{seed}.npz") as a:
                    base = np.clip(a["test_probabilities"], 1e-10, 1 - 1e-10)
                    np.testing.assert_array_equal(ids, a["test_ids"])
                    np.testing.assert_array_equal(target, a["test_targets"])
            else:
                base = reference_p
                np.testing.assert_array_equal(ids, test)
            bits = ((target[:, None] >> np.arange(10)) & 1).astype(float)

            def loss(z, target_bits=bits):
                return -(target_bits * np.log(z) + (1 - target_bits) * np.log1p(-z)).mean(axis=1)

            def correct(z, target_bits=bits):
                return ((z >= 0.5) == target_bits).all(axis=1).astype(float)

            bce.append(loss(p) - loss(base))
            accuracy.append(correct(p) - correct(base))
        _, inverse = np.unique(groups, return_inverse=True)
        counts = np.bincount(inverse)
        rng = np.random.default_rng(12345)
        draws = rng.integers(0, len(counts), size=(2000, len(counts)))
        row_result = {}
        for label, values in [("bce_difference", bce), ("exact_accuracy_difference", accuracy)]:
            average = np.mean(values, axis=0)
            sums = np.bincount(inverse, weights=average)
            boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            row_result[label] = {
                "mean": float(average.mean()),
                "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            }
        result[method] = row_result
    return result


def table(summary, partition):
    rows = [
        "| Encoding | BCE ↓ (mean ± SD) | Exact word ↑ | Top-5 ↑ | Vocabulary NLL ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in sorted(summary.items(), key=lambda item: item[1][partition]["bce"]["mean"]):
        m = r[partition]
        rows.append(
            f"| `{name}` | {m['bce']['mean']:.4f} ± {m['bce']['sd']:.4f} "
            f"| {100 * m['exact_word_accuracy']['mean']:.2f}% "
            f"| {100 * m['vocab_top5']['mean']:.2f}% | {m['vocab_nll']['mean']:.3f} |"
        )
    return "\n".join(rows)


def main():
    screen, confirmation = aggregate("screen"), aggregate("confirmation")
    if len(screen) != 26 or len(confirmation) != 8:
        raise ValueError("The planned sweep is not complete")
    if any(len(r["seeds"]) != 3 for stage in [screen, confirmation] for r in stage.values()):
        raise ValueError("Missing seed runs")
    intervals = paired_intervals(confirmation)
    controls = {
        name: paired_intervals(confirmation, name)
        for name in ("bit_prior", "bigram_bits", "constant_zero")
    }
    write_json(
        ROOT / "results/summary.json",
        {
            "screen": screen,
            "confirmation": confirmation,
            "paired_test_intervals": intervals,
            "control_test_intervals": controls,
        },
    )
    (ROOT / "results/screen_table.md").write_text(table(screen, "validation") + "\n")
    (ROOT / "results/confirmation_table.md").write_text(table(confirmation, "test") + "\n")
    baselines = json.loads((ROOT / "results/confirmation/baselines.json").read_text())["test"]
    ordered = sorted(confirmation, key=lambda m: confirmation[m]["test"]["bce"]["mean"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    colors = ["#8a98a5" if m == "constant_zero" else "#235ea8" for m in ordered]
    for ax, metric, label in zip(
        axes,
        ["bce", "exact_word_accuracy"],
        ["Held-out bitwise BCE (lower is better)", "Exact next-word accuracy (%)"],
        strict=True,
    ):
        scale = 100 if metric == "exact_word_accuracy" else 1
        values = [confirmation[m]["test"][metric]["mean"] * scale for m in ordered]
        errors = [confirmation[m]["test"][metric]["sd"] * scale for m in ordered]
        if metric == "bce":
            positions = np.arange(len(ordered))
            ax.errorbar(values, positions, xerr=errors, fmt="none", ecolor="#66788a", capsize=3)
            ax.scatter(values, positions, c=colors, s=45)
            ax.set_yticks(positions, ordered)
        else:
            ax.barh(ordered, values, xerr=errors, color=colors, alpha=0.9, capsize=3)
        ax.axvline(
            baselines["bit_prior"][metric] * scale,
            color="#b86522",
            ls="--",
            label="Classical bit prior",
        )
        ax.axvline(
            baselines["bigram_bits"][metric] * scale,
            color="#21865d",
            ls=":",
            label="Classical bigram bits / direct RY loading",
        )
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "bce":
            ax.set_xlim(0.48, max(values) + 0.025)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle(
        "QCSE causal encoding confirmation · 3 seeds · 1,246 held-out targets", fontsize=13
    )
    fig.savefig(ROOT / "results/confirmation.png", dpi=180)
    plt.close(fig)
    print(table(confirmation, "test"))
    print(json.dumps(intervals, indent=2))


if __name__ == "__main__":
    main()
