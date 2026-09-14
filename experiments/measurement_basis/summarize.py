"""Aggregate the learned measurement-basis experiment and paired comparisons."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qcse.data import load_phrases

ROOT = Path("experiments/measurement_basis")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def aggregate(stage):
    grouped = {}
    for path in (ROOT / "results" / stage).glob("*_seed*.json"):
        row = json.loads(path.read_text())
        grouped.setdefault(row["method"], []).append(row)
    result = {}
    for method, rows in grouped.items():
        rows.sort(key=lambda row: row["seed"])
        summary = {
            "seeds": [row["seed"] for row in rows],
            "parameters": rows[0]["parameters"],
            "ansatz_parameters": rows[0]["ansatz_parameters"],
            "readout_parameters": rows[0]["readout_parameters"],
            "selected_epochs": [row["selected_epoch"] for row in rows],
            "seconds_total": sum(row["seconds"] for row in rows),
        }
        for partition in rows[0]["metrics"]:
            summary[partition] = {
                metric: {
                    "mean": float(np.mean([row["metrics"][partition][metric] for row in rows])),
                    "sd": float(
                        np.std([row["metrics"][partition][metric] for row in rows], ddof=1)
                    ),
                }
                for metric in rows[0]["metrics"][partition]
            }
            summary[partition]["readout_reset_bce_delta"] = {
                "mean": float(
                    np.mean(
                        [row["diagnostics"][partition]["readout_reset_bce_delta"] for row in rows]
                    )
                ),
                "sd": float(
                    np.std(
                        [row["diagnostics"][partition]["readout_reset_bce_delta"] for row in rows],
                        ddof=1,
                    )
                ),
            }
            summary[partition]["last_diagonal_ablation_bce_delta"] = {
                "mean": float(
                    np.mean(
                        [
                            row["diagnostics"][partition][
                                "last_diagonal_ablation_bce_delta"
                            ]
                            for row in rows
                        ]
                    )
                ),
                "sd": float(
                    np.std(
                        [
                            row["diagnostics"][partition][
                                "last_diagonal_ablation_bce_delta"
                            ]
                            for row in rows
                        ],
                        ddof=1,
                    )
                ),
            }
        summary["mean_tilt_degrees"] = {
            "mean": float(np.mean([row["readout"]["mean_tilt_degrees"] for row in rows])),
            "sd": float(
                np.std([row["readout"]["mean_tilt_degrees"] for row in rows], ddof=1)
            ),
        }
        result[method] = summary
    return result


def example_metrics(probabilities, targets, vocabulary_size):
    qubits = probabilities.shape[1]
    bits = ((targets[:, None] >> np.arange(qubits)) & 1).astype(float)
    p = np.clip(probabilities, 1e-10, 1 - 1e-10)
    vocabulary_bits = (
        (np.arange(vocabulary_size)[:, None] >> np.arange(qubits)) & 1
    ).astype(float)
    scores = np.log(p) @ vocabulary_bits.T + np.log1p(-p) @ (1 - vocabulary_bits).T
    row_max = scores.max(axis=1, keepdims=True)
    log_z = row_max[:, 0] + np.log(np.exp(scores - row_max).sum(axis=1))
    ranking = np.argsort(-scores, axis=1, kind="stable")
    return {
        "bce": -(bits * np.log(p) + (1 - bits) * np.log1p(-p)).mean(axis=1),
        "exact_word_accuracy": ((p >= 0.5) == bits).all(axis=1).astype(float),
        "vocab_top5": np.any(ranking[:, :5] == targets[:, None], axis=1).astype(float),
        "vocab_nll": log_z - scores[np.arange(len(targets)), targets],
    }


def paired_intervals():
    directory = ROOT / "results" / "confirmation"
    learned_rows = sorted(directory.glob("learned_xyz_seed*.npz"))
    if len(learned_rows) != 3:
        raise ValueError("Expected three learned_xyz confirmation runs")
    sentences = load_phrases()
    sentence_keys = {tuple(sentence): index for index, sentence in enumerate(sentences)}
    differences = {}
    groups = None
    seed_means = {}
    for learned_path in learned_rows:
        seed = learned_path.stem.rsplit("seed", 1)[1]
        with np.load(learned_path) as learned, np.load(directory / f"z_seed{seed}.npz") as baseline:
            np.testing.assert_array_equal(learned["test_ids"], baseline["test_ids"])
            np.testing.assert_array_equal(learned["test_targets"], baseline["test_targets"])
            target = learned["test_targets"]
            vocabulary_size = len(learned["vocabulary"])
            learned_metrics = example_metrics(
                learned["test_probabilities"], target, vocabulary_size
            )
            baseline_metrics = example_metrics(
                baseline["test_probabilities"], target, vocabulary_size
            )
            current_groups = np.array(
                [sentence_keys[tuple(sentences[i])] for i in learned["test_sentence_ids"]]
            )
            if groups is None:
                groups = current_groups
            else:
                np.testing.assert_array_equal(groups, current_groups)
            for metric in learned_metrics:
                differences.setdefault(metric, []).append(
                    learned_metrics[metric] - baseline_metrics[metric]
                )
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(20260914)
    draws = rng.integers(0, len(counts), size=(5000, len(counts)))
    result = {}
    for metric, rows in differences.items():
        rows = np.asarray(rows)
        average = rows.mean(axis=0)
        sums = np.bincount(inverse, weights=average)
        bootstrap = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        seed_means[metric] = rows.mean(axis=1).tolist()
        result[metric] = {
            "mean": float(average.mean()),
            "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "per_seed_means": seed_means[metric],
        }
    return result


def table(summary, partition):
    rows = [
        "| Readout | Parameters | BCE ↓ | Exact word ↑ | Top-5 ↑ | Vocabulary NLL ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("z", "learned_xyz"):
        row = summary[method]
        metrics = row[partition]
        rows.append(
            f"| `{method}` | {row['parameters']} "
            f"| {metrics['bce']['mean']:.4f} ± {metrics['bce']['sd']:.4f} "
            f"| {100 * metrics['exact_word_accuracy']['mean']:.2f}% "
            f"| {100 * metrics['vocab_top5']['mean']:.2f}% "
            f"| {metrics['vocab_nll']['mean']:.3f} |"
        )
    return "\n".join(rows)


def plot(summary):
    methods = ["z", "learned_xyz"]
    labels = ["Z basis", "Learned arbitrary axis"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    for ax, metric, title, scale in zip(
        axes,
        ("bce", "exact_word_accuracy", "vocab_top5"),
        ("Bitwise BCE (lower is better)", "Exact next word (%)", "Top-5 next word (%)"),
        (1, 100, 100),
        strict=True,
    ):
        values = [summary[method]["test"][metric]["mean"] * scale for method in methods]
        errors = [summary[method]["test"][metric]["sd"] * scale for method in methods]
        bars = ax.bar(labels, values, yerr=errors, color=["#7b8794", "#2667a8"], capsize=4)
        ax.bar_label(bars, fmt="%.3f" if metric == "bce" else "%.2f", padding=5)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelrotation=12)
        if metric == "bce":
            lower = min(value - error for value, error in zip(values, errors, strict=True))
            upper = max(value + error for value, error in zip(values, errors, strict=True))
            margin = max(0.001, 0.5 * (upper - lower))
            ax.set_ylim(lower - margin, upper + margin)
    fig.suptitle("Causal QCSE: window 8, 8 layers, 3 seeds, 1,246 held-out targets", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(ROOT / "results" / "comparison.png", dpi=180)
    plt.close(fig)


def main():
    screen = aggregate("screen")
    confirmation = aggregate("confirmation")
    if set(screen) != {"z", "learned_y", "learned_xyz"}:
        raise ValueError("Screen is incomplete")
    if set(confirmation) != {"z", "learned_xyz"}:
        raise ValueError("Confirmation is incomplete")
    paired = paired_intervals()
    result = {"screen": screen, "confirmation": confirmation, "paired_test": paired}
    write_json(ROOT / "results" / "summary.json", result)
    (ROOT / "results" / "comparison_table.md").write_text(table(confirmation, "test") + "\n")
    plot(confirmation)
    print(table(confirmation, "test"))
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
