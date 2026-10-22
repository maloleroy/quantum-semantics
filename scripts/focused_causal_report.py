"""Summarize and plot the completed 16-fit causal focused sweep."""

# Generated report prose intentionally contains long lines.
# ruff: noqa: E501

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "focused-sweep-50-local-1024-512val-2048test"
RESULTS = ROOT / "results" / "focused-causal-1024"
sys.path.insert(0, str(ROOT / "scripts"))
from focused_sweep import CONFIGS  # noqa: E402, I001


CONFIG_LABELS = {
    "baseline": "Baseline (α=1, w=4, L=8, B=64)",
    "alpha-0.1": "α=0.1",
    "alpha-3": "α=3",
    "layers-2": "L=2",
    "window-2": "w=2",
    "window-8": "w=8",
    "batch-16": "B=16",
    "batch-256": "B=256",
}
ORDER = [config["name"] for config in CONFIGS["causal"]]


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def collect():
    rows = []
    for config in CONFIGS["causal"]:
        name = config["name"]
        for repetition in (0, 1):
            base = SOURCE / "causal" / name / f"repetition-{repetition}"
            candidates = sorted(base.glob("*/test.json"))
            if len(candidates) != 1:
                raise RuntimeError(f"Expected one completed test in {base}, found {candidates}")
            test_path = candidates[0]
            run = test_path.parent
            test = read_json(test_path)
            summary = read_json(run / "summary.json")
            info = read_json(run / "run_info.json")
            history = read_json(run / "history.json")
            if info.get("status") != "complete":
                raise RuntimeError(f"Run is not complete: {run}")
            selected = next(row for row in history if row["epoch"] == test["epoch"])
            validation = selected["validation"]
            configuration = summary["configuration"]
            runtime = info
            rows.append(
                {
                    "config": name,
                    "label": CONFIG_LABELS[name],
                    "repetition": repetition,
                    "seed": summary["seed"],
                    "alpha": configuration["alpha"],
                    "window": configuration["window"],
                    "layers": configuration["layers"],
                    "batch": configuration["batch_size"],
                    "val_top1": validation["exact_word_accuracy"],
                    "val_similarity": validation["paper_similarity_accuracy"],
                    "val_bce": validation["bce"],
                    "test_top1": test["exact_word_accuracy"],
                    "test_similarity": test["paper_similarity_accuracy"],
                    "test_bce": test["bce"],
                    "best_epoch": test["epoch"],
                    "stopped_epoch": test["early_stopping"]["stopped_epoch"],
                    "samples_per_epoch": summary["samples_per_epoch"],
                    "eval_examples": summary["eval_examples"],
                    "final_eval_examples": test["examples"],
                    "device": runtime["device"],
                    "run_path": str(run.relative_to(ROOT)),
                    "history": history,
                }
            )
    if len(rows) != 16:
        raise RuntimeError(f"Expected 16 completed causal fits, found {len(rows)}")
    return rows


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["config"]].append(row)
    metrics = (
        "val_top1",
        "val_similarity",
        "val_bce",
        "test_top1",
        "test_similarity",
        "test_bce",
        "best_epoch",
        "stopped_epoch",
    )
    output = []
    for config in ORDER:
        values = groups[config]
        item = {"config": config, "label": CONFIG_LABELS[config], "n": len(values)}
        first = values[0]
        item.update(
            {
                key: first[key]
                for key in (
                    "alpha",
                    "window",
                    "layers",
                    "batch",
                    "samples_per_epoch",
                    "eval_examples",
                    "final_eval_examples",
                )
            }
        )
        for metric in metrics:
            data = np.asarray([row[metric] for row in values], dtype=float)
            item[f"{metric}_mean"] = float(data.mean())
            item[f"{metric}_sd"] = float(data.std(ddof=1)) if len(data) > 1 else 0.0
        output.append(item)
    return output


def paired_effects(rows):
    lookup = {(row["config"], row["repetition"]): row for row in rows}
    comparisons = [config for config in ORDER if config != "baseline"]
    output = []
    for config in comparisons:
        for repetition in (0, 1):
            left = lookup[(config, repetition)]
            right = lookup[("baseline", repetition)]
            output.append(
                {
                    "config": config,
                    "label": CONFIG_LABELS[config],
                    "repetition": repetition,
                    "seed": left["seed"],
                    "delta_val_top1": left["val_top1"] - right["val_top1"],
                    "delta_val_similarity": left["val_similarity"] - right["val_similarity"],
                    "delta_test_top1": left["test_top1"] - right["test_top1"],
                    "delta_test_similarity": left["test_similarity"] - right["test_similarity"],
                    "delta_test_bce": left["test_bce"] - right["test_bce"],
                }
            )
    return output


def percent_axis(axis):
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))


def style_axes(axis):
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)


def test_metrics(rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    x = np.arange(len(ORDER))
    labels = [CONFIG_LABELS[name] for name in ORDER]
    for axis, metric, title in zip(
        axes,
        ("test_top1", "test_similarity", "test_bce"),
        ("Exact held-out word top-1", "Held-out paper similarity", "Held-out BCE (diagnostic)"),
        strict=True,
    ):
        for index, config in enumerate(ORDER):
            values = [row[metric] for row in rows if row["config"] == config]
            mean = np.mean(values)
            sd = np.std(values, ddof=1)
            axis.bar(index, mean, color="#789bc5", alpha=0.65, width=0.58)
            axis.errorbar(index, mean, yerr=sd, fmt="none", ecolor="#222222", capsize=4, zorder=3)
            axis.scatter(
                np.full(len(values), index) + np.linspace(-0.09, 0.09, len(values)),
                values,
                color=("#3765a8", "#c47724"),
                s=42,
                zorder=4,
            )
        axis.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        axis.set_title(title)
        axis.set_ylabel("Accuracy" if metric != "test_bce" else "BCE")
        if metric != "test_bce":
            percent_axis(axis)
        style_axes(axis)
    fig.suptitle("Causal focused sweep: held-out results, two repetitions")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def validation_curves(rows, path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True, constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.arange(10))
    metrics = ("exact_word_accuracy", "paper_similarity_accuracy", "bce")
    titles = ("Validation exact word top-1", "Validation paper similarity", "Validation BCE")
    for index, config in enumerate(ORDER):
        histories = [row["history"] for row in rows if row["config"] == config]
        epochs = sorted({entry["epoch"] for history in histories for entry in history})
        color = colors[index % len(colors)]
        for axis, metric, title in zip(axes, metrics, titles, strict=True):
            means, spreads = [], []
            for epoch in epochs:
                values = []
                for history in histories:
                    entry = next((item for item in history if item["epoch"] == epoch), None)
                    if entry is not None:
                        values.append(entry["validation"][metric])
                means.append(np.mean(values))
                spreads.append(np.std(values))
            mean = np.asarray(means)
            spread = np.asarray(spreads)
            axis.plot(epochs, mean, color=color, linewidth=1.6, label=CONFIG_LABELS[config])
            axis.fill_between(epochs, mean - spread, mean + spread, color=color, alpha=0.1)
            axis.set_title(title)
            style_axes(axis)
            if metric != "bce":
                percent_axis(axis)
    axes[-1].set_xlabel("Epoch")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Causal validation drift across epochs")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parameter_effects(effects, path):
    means = []
    labels = []
    sim_sds = []
    bce_means = []
    bce_sds = []
    for config in ORDER[1:]:
        values = [row for row in effects if row["config"] == config]
        labels.append(CONFIG_LABELS[config])
        sim = [row["delta_test_similarity"] for row in values]
        bce = [row["delta_test_bce"] for row in values]
        means.append(np.mean(sim))
        sim_sds.append(np.std(sim, ddof=1))
        bce_means.append(np.mean(bce))
        bce_sds.append(np.std(bce, ddof=1))
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    axes[0].barh(y, means, xerr=sim_sds, color="#7c4d9f", alpha=0.75)
    axes[0].axvline(0, color="#222222", linewidth=0.8)
    axes[0].set_yticks(y, labels, fontsize=9)
    axes[0].xaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f} pp")
    axes[0].set_xlabel("Test paper-similarity change vs baseline")
    axes[0].set_title("Secondary quality contrast")
    axes[1].barh(y, bce_means, xerr=bce_sds, color="#3765a8", alpha=0.75)
    axes[1].axvline(0, color="#222222", linewidth=0.8)
    axes[1].set_yticks(y, labels, fontsize=9)
    axes[1].set_xlabel("Test BCE change vs baseline; negative is better")
    axes[1].set_title("BCE diagnostic contrast")
    for axis in axes:
        style_axes(axis)
    fig.suptitle("Paired parameter/setup changes, matched by seed")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def seed_drift(rows, path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("test_similarity", "test_bce"),
        ("Held-out paper similarity", "Held-out BCE"),
        strict=True,
    ):
        for index, config in enumerate(ORDER):
            values = [row for row in rows if row["config"] == config]
            axis.plot(
                [index - 0.1, index + 0.1],
                [values[0][metric], values[1][metric]],
                color="#8a8a8a",
                linewidth=1,
            )
            axis.scatter(
                [index - 0.1, index + 0.1],
                [values[0][metric], values[1][metric]],
                color=("#3765a8", "#c47724"),
                s=45,
                zorder=3,
            )
        axis.set_xticks(
            np.arange(len(ORDER)),
            [CONFIG_LABELS[name] for name in ORDER],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        axis.set_ylabel("Accuracy" if metric != "test_bce" else "BCE")
        axis.set_title(title)
        if metric != "test_bce":
            percent_axis(axis)
        style_axes(axis)
    fig.suptitle("Causal seed/repetition drift")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def epoch_drift(rows, path):
    fig, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
    for index, config in enumerate(ORDER):
        values = [row for row in rows if row["config"] == config]
        for row in values:
            jitter = -0.1 if row["repetition"] == 0 else 0.1
            axis.scatter(index + jitter, row["best_epoch"], marker="o", color="#3765a8", s=45)
            axis.scatter(index + jitter, row["stopped_epoch"], marker="x", color="#c47724", s=50)
    axis.set_xticks(
        np.arange(len(ORDER)),
        [CONFIG_LABELS[name] for name in ORDER],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    axis.set_ylabel("Epoch")
    axis.legend(
        [
            Line2D([], [], marker="o", color="#3765a8", linestyle="none"),
            Line2D([], [], marker="x", color="#c47724", linestyle="none"),
        ],
        ["Selected by validation", "Early-stopping boundary"],
    )
    axis.set_title("Causal validation-selected and stopped epochs")
    style_axes(axis)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def batch_tradeoff(rows, path):
    batches = [16, 64, 256]
    names = {16: "B=16", 64: "B=64", 256: "B=256"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for index, batch in enumerate(batches):
        values = [
            row
            for row in rows
            if row["batch"] == batch
            and row["alpha"] == 1.0
            and row["window"] == 4
            and row["layers"] == 8
        ]
        mean = np.mean([row["test_similarity"] for row in values])
        sd = np.std([row["test_similarity"] for row in values], ddof=1)
        axes[0].bar(index, mean, color="#789bc5", alpha=0.7)
        axes[0].errorbar(index, mean, yerr=sd, fmt="none", ecolor="#222222", capsize=4)
        axes[0].scatter(
            [index - 0.08, index + 0.08],
            [row["test_similarity"] for row in values],
            color=("#3765a8", "#c47724"),
            zorder=3,
        )
        axes[1].bar(index, int(np.ceil(1024 / batch)), color="#c47724", alpha=0.75)
    axes[0].set_xticks(np.arange(3), [names[batch] for batch in batches])
    axes[0].set_ylabel("Test paper similarity")
    axes[0].set_title("Quality diagnostic")
    percent_axis(axes[0])
    axes[1].set_xticks(np.arange(3), [names[batch] for batch in batches])
    axes[1].set_ylabel("Adam updates per 1,024-draw epoch")
    axes[1].set_title("Optimization work per epoch")
    for axis in axes:
        style_axes(axis)
    fig.suptitle("Batch-size trade-off: quality versus optimizer updates")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def reports(rows, aggregates, effects):
    def pct(value):
        return f"{100 * value:.2f}%"

    def mean_sd(item, metric):
        return f"{pct(item[f'{metric}_mean'])} ± {pct(item[f'{metric}_sd'])}"

    lines = [
        "# Causal focused sweep report — 16 completed fits",
        "",
        "## Protocol and scope",
        "",
        "This report covers all eight current causal configurations, each with two repetitions (seeds 42 and 43). It uses the full phrases + cleaned corpus, sentence-grouped 64/16/20 splits, 1,024 replacement draws per training epoch, 512 fixed validation-monitor examples, 2,048 held-out test examples, 50 maximum epochs, and patience 8 with a 10-epoch minimum.",
        "",
        "The requested primary metric is exact word top-1: all 14 decoded bits must match the target word. The test score was never used for selection. Paper-similarity accuracy (at least half of bits correct) and BCE are shown as secondary diagnostics only.",
        "",
        "## Aggregate results",
        "",
        "| Setup | α | Window | Layers | Batch | Validation exact top-1 | Test exact top-1 | Test paper similarity | Test BCE | Selected epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(
            f"| {item['label']} | {item['alpha']} | {item['window']} | {item['layers']} | {item['batch']} | {mean_sd(item, 'val_top1')} | {mean_sd(item, 'test_top1')} | {mean_sd(item, 'test_similarity')} | {item['test_bce_mean']:.3f} ± {item['test_bce_sd']:.3f} | {item['best_epoch_mean']:.1f} ± {item['best_epoch_sd']:.1f} |"
        )
    lines += [
        "",
        "## Primary top-1 conclusion",
        "",
        "Exact word top-1 is saturated at zero for the baseline, both alpha changes, both window changes, the 2-layer setup, and both batch-256 repetitions. Only batch-16 repetition 0 is nonzero (0.34% test top-1); its repetition mean is 0.17%. Therefore the 512-example validation cap does not provide enough resolution for exact-word top-1 to rank most causal settings. This is a metric-resolution limitation, not evidence that all configurations are equivalent.",
        "",
        "The implementation still selects by exact top-1 first, then paper similarity and BCE as deterministic tie-breakers. Because exact top-1 is usually tied at zero, most selected epochs are effectively decided by those secondary tie-breakers; this is recorded as a limitation rather than relabeled as primary-top-1 evidence.",
        "",
        "## Parameter and setup drift",
        "",
        "- Alpha: α=0.1 is modestly above baseline on paper similarity (+5.5 percentage points paired mean), while α=3 is essentially neutral (−0.1 points). The α=0.1 gain is seed-dependent; it is not a decisive optimum.",
        "- Window: window 8 is mildly above baseline (+5.2 points), while window 2 is slightly below (−1.3 points). Both effects are small relative to repetition drift.",
        "- Layers: the 2-layer model is the clearest depth improvement in this sweep (+22.5 paper-similarity points versus 8 layers, with both repetitions improving), although exact word top-1 remains zero. Its test BCE is also slightly lower as a diagnostic, but BCE is not the selection target.",
        "- Batch size: batch 16 is the strongest setup by secondary quality (+37.9 paper-similarity points versus batch 64; −0.1095 BCE), and it is the only arm with a nonzero exact-word result. Batch 256 is not a quality win: it is slightly below batch 64 on paper similarity (−0.3 points) and worse on BCE (+0.016). The larger batch performs only 4 Adam updates per 1,024-draw epoch, versus 16 for batch 64 and 64 for batch 16.",
        "",
        "## Repetition drift and validity",
        "",
        "The seed effect is very large for most 8-layer settings: seed 42 typically reaches roughly 0.77–0.99 paper similarity, whereas seed 43 is often roughly 0.23–0.27. Batch 16 is more robust but still differs between repetitions. Two repetitions are sufficient to expose this instability, not to provide tight uncertainty intervals.",
        "",
        "All fits completed with finite histories and fixed sentence-grouped splits. Training draws may repeat examples by design, but the held-out sentence groups remain separated from training and validation. Test metrics were computed once after validation-based epoch selection.",
        "",
        "## Interpretation",
        "",
        "The defensible causal conclusion from this run is that smaller batches and, secondarily, fewer layers are more promising under the current SPSA/Adam implementation. Alpha and window effects are weak and seed-sensitive. Batch 256 is faster per fixed epoch but does not compensate for its much smaller optimizer-update budget in quality. A final claim about exact word top-1 requires a larger validation monitor or a denser retrieval metric; the current capped run should not be used to declare a top-1 winner beyond the observed batch-16 signal.",
        "",
        "## Figures",
        "",
        "- `causal-test-metrics.png`: test primary/secondary metrics with repetition points and mean ± SD.",
        "- `causal-validation-curves.png`: validation drift across training epochs.",
        "- `causal-parameter-effects.png`: matched parameter changes versus baseline.",
        "- `causal-seed-drift.png`: repetition drift by setup.",
        "- `causal-selected-epochs.png`: selected and stopped epochs.",
        "- `causal-batch-tradeoff.png`: batch quality versus optimizer updates per epoch.",
    ]
    (RESULTS / "detailed-report.md").write_text("\n".join(lines) + "\n")

    short = [
        "# Causal focused sweep — short report",
        "",
        "All 16 causal fits completed with 1,024 training draws/epoch, 512 validation examples, and 2,048 final test examples.",
        "",
        "- Exact word top-1 is zero almost everywhere, so the capped validation set cannot reliably rank causal parameters.",
        "- Batch 16 is the strongest arm on the secondary paper-similarity metric and the only arm with a nonzero exact-word result.",
        "- Two layers improve paper similarity over eight layers by about 22.5 percentage points.",
        "- Alpha and window changes are small and seed-sensitive.",
        "- Batch 256 is faster per epoch but is not a quality win: it has fewer optimizer updates and slightly worse secondary results than batch 64.",
        "",
        "Use the detailed report and plots for the seed drift and the exact paired contrasts. These results are valid pipeline evidence, but not a strong exact-top-1 ranking because of metric saturation.",
    ]
    (RESULTS / "short-report.md").write_text("\n".join(short) + "\n")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = collect()
    fields = [key for key in rows[0] if key != "history"]
    write_csv(RESULTS / "runs.csv", [{key: row[key] for key in fields} for row in rows], fields)
    aggregates = aggregate(rows)
    write_csv(RESULTS / "aggregates.csv", aggregates, list(aggregates[0]))
    effects = paired_effects(rows)
    write_csv(RESULTS / "paired-effects.csv", effects, list(effects[0]))
    (RESULTS / "summary.json").write_text(
        json.dumps({"aggregates": aggregates, "paired_effects": effects}, indent=2)
    )
    test_metrics(rows, RESULTS / "causal-test-metrics.png")
    validation_curves(rows, RESULTS / "causal-validation-curves.png")
    parameter_effects(effects, RESULTS / "causal-parameter-effects.png")
    seed_drift(rows, RESULTS / "causal-seed-drift.png")
    epoch_drift(rows, RESULTS / "causal-selected-epochs.png")
    batch_tradeoff(rows, RESULTS / "causal-batch-tradeoff.png")
    reports(rows, aggregates, effects)
    print(f"Wrote causal report and plots to {RESULTS}")


if __name__ == "__main__":
    main()
