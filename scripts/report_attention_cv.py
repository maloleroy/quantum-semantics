"""Create plots and a Markdown report from attention CV results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    results = json.loads((args.root / "results.json").read_text())
    output = args.report_output or args.root
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    rows = results["results"]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for row in rows:
        histories = [
            json.loads(
                (args.root / row["setup"]["name"] / f"fold-{fold:02d}" / "history.json").read_text()
            )
            for fold in (1, 2)
        ]
        epochs = [item["epoch"] for item in histories[0]]
        mean = np.mean(
            [[item["validation"]["cross_entropy"] for item in history] for history in histories],
            axis=0,
        )
        std = np.std(
            [[item["validation"]["cross_entropy"] for item in history] for history in histories],
            axis=0,
            ddof=1,
        )
        axes[0].plot(epochs, mean, label=row["setup"]["name"])
        axes[0].fill_between(epochs, mean - std, mean + std, alpha=0.12)
        axes[1].plot(
            epochs,
            [item["validation"]["cosine_top5"] for item in histories[0]],
            label=row["setup"]["name"],
        )
    axes[0].set(title="Validation cross-entropy", xlabel="epoch", ylabel="cross-entropy")
    axes[1].set(title="Validation cosine top-5 (fold 1)", xlabel="epoch", ylabel="top-5")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output / "validation-curves.png", dpi=180)
    plt.close(figure)

    labels = [row["setup"]["name"] for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("cross_entropy", "cosine_top1", "cosine_top5"),
        ("test cross-entropy", "test cosine top-1", "test cosine top-5"),
        strict=True,
    ):
        values = [row["test"][metric] for row in rows]
        axis.bar(x, values, color=["#4472c4" if "qcse" in label else "#ed7d31" for label in labels])
        upper = max(values) * (1.2 if metric != "cross_entropy" else 1.05)
        axis.set(title=title, xticks=x, xticklabels=labels, ylim=(0, max(upper, 1e-3)))
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
    figure.savefig(output / "test-comparison.png", dpi=180)
    plt.close(figure)

    settings = results["settings"]
    report = [
        "# Quantum attention 25-epoch cross-validation",
        "",
        (
            "This report compares fixed QCSE and learned classical input encodings feeding the "
            "same causal quantum attention model. The corpus uses both `phrases` and `cleaned`, "
            "balanced sentence sampling, and a shared vocabulary."
        ),
        "",
        (
            f"The run used {settings['sentences']:,} sentences, {settings['examples']:,} token "
            f"examples, {settings['vocabulary_size']:,} vocabulary words, {settings['epochs']} "
            f"epochs per fit, {settings['samples_per_epoch']:,} replacement draws per epoch, "
            f"and a {settings['development_examples']:,}/{settings['test_examples']:,} "
            "development/test example split. Each setup used two independent 20% validation "
            "shuffle splits inside development, then a fresh refit on all development examples "
            "before one held-out test score."
        ),
        "",
        "![Validation curves](validation-curves.png)",
        "",
        "![Test comparison](test-comparison.png)",
        "",
        (
            "| Setup | CV validation CE mean ± SD | CV cosine top-1 | CV cosine top-5 | "
            "Test CE | Test cosine top-1 | Test cosine top-5 |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        validation = row["validation"]
        test = row["test"]
        report.append(
            f"| {row['setup']['name']} | {validation['cross_entropy_mean']:.4f} ± "
            f"{validation['cross_entropy_std']:.4f} | "
            f"{validation['cosine_top1_mean']:.3%} | "
            f"{validation['cosine_top5_mean']:.3%} | {test['cross_entropy']:.4f} | "
            f"{test['cosine_top1']:.3%} | {test['cosine_top5']:.3%} |"
        )
    best = min(rows, key=lambda row: row["validation"]["cross_entropy_mean"])
    report.extend(
        [
            "",
            (
                f"The lowest mean validation cross-entropy was `{best['setup']['name']}`. "
                "Superiority should be judged from the repeated validation mean and spread; "
                "the single held-out test score is reported for confirmation and was not used "
                "to select a setup."
            ),
            "",
            (
                "The experiment is a statevector simulation of the paper's bilinear attention "
                "adaptation. It uses real overlaps, causal masking and normalized postselection; "
                "it does not claim finite-shot hardware performance or an efficient block-encoding "
                "oracle. The local run used a balanced 5,000-sentence cap so all 1,000 phrase rows "
                "were retained alongside cleaned sentences. Omit `--max-sentences` on a GPU "
                "allocation to use the full curated pool."
            ),
            "",
            (
                "Generated files: `results.json`, per-fold histories/checkpoints, "
                "`validation-curves.png`, and `test-comparison.png`."
            ),
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote plots and report under {output}")


if __name__ == "__main__":
    main()
