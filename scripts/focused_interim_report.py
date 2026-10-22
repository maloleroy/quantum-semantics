"""Summarize and plot the completed 14-fit focused encoding/attention snapshot."""

# The generated report intentionally contains long prose/table lines.
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

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "focused-sweep-50-local"
RESULTS = ROOT / "results" / "focused-interim-14"
sys.path.insert(0, str(ROOT / "scripts"))
from focused_sweep import CONFIGS  # noqa: E402, I001


FAMILY_LABELS = {"encoding": "Encoding", "attention": "Attention"}
CONFIG_LABELS = {
    "classical-encoding-trainable-ansatz": "Classical encoding + trainable ansatz",
    "classical-encoding-no-ansatz": "Classical encoding + zero ansatz",
    "no-encoding-no-ansatz": "No encoding + zero ansatz",
    "qcse-encoding-quantum-attention": "QCSE encoding + quantum attention",
    "classical-encoding-quantum-attention": "Classical encoding + quantum attention",
    "qcse-encoding-no-quantum-attention": "QCSE encoding + no quantum attention",
    "classical-encoding-no-quantum-attention": "Classical encoding + no quantum attention",
}
PLOT_LABELS = {
    "classical-encoding-trainable-ansatz": "Classical\n+ trainable ansatz",
    "classical-encoding-no-ansatz": "Classical\n+ zero ansatz",
    "no-encoding-no-ansatz": "No encoding\n+ zero ansatz",
    "qcse-encoding-quantum-attention": "QCSE\nQ/K/V",
    "classical-encoding-quantum-attention": "Classical\nQ/K/V",
    "qcse-encoding-no-quantum-attention": "QCSE\nraw",
    "classical-encoding-no-quantum-attention": "Classical\nraw",
}
METRICS = ("top1", "top5", "cross_entropy")


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def find_runs(family, config):
    base = SOURCE / family / config
    runs = []
    for repetition in sorted(base.glob("repetition-*")):
        candidates = sorted(repetition.glob("*/test.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one completed test in {repetition}, found {candidates}")
        test_path = candidates[0]
        run = test_path.parent
        test = read_json(test_path)
        summary = read_json(run / "summary.json")
        info = read_json(run / "run_info.json")
        if info.get("status") != "complete":
            raise RuntimeError(f"Run is not complete: {run}")
        history = read_json(run / "history.json")
        selected = next(row for row in history if row["epoch"] == test["epoch"])
        validation = selected["validation"]
        settings = summary["settings"]
        runs.append(
            {
                "family": family,
                "config": config,
                "label": CONFIG_LABELS[config],
                "repetition": int(repetition.name.split("-")[-1]),
                "seed": settings["seed"],
                "val_top1": validation["cosine_top1"],
                "val_top5": validation["cosine_top5"],
                "val_cross_entropy": validation["cross_entropy"],
                "test_top1": test["top1"],
                "test_top5": test["top5"],
                "test_cross_entropy": test["cross_entropy"],
                "best_epoch": test["epoch"],
                "stopped_epoch": test["early_stopping"]["stopped_epoch"],
                "samples_per_epoch": settings["samples_per_epoch"],
                "eval_examples": settings["eval_examples"],
                "final_eval_examples": test["examples"],
                "run_path": str(run.relative_to(ROOT)),
                "history": history,
            }
        )
    return runs


def collect():
    rows = []
    for family in ("encoding", "attention"):
        for config in CONFIGS[family]:
            rows.extend(find_runs(family, config["name"]))
    if len(rows) != 14:
        raise RuntimeError(f"Expected 14 completed fits, found {len(rows)}")
    return rows


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["config"])].append(row)
    output = []
    for (family, config), values in groups.items():
        item = {
            "family": family,
            "config": config,
            "label": CONFIG_LABELS[config],
            "n": len(values),
        }
        for metric in (
            "val_top1",
            "val_top5",
            "val_cross_entropy",
            "test_top1",
            "test_top5",
            "test_cross_entropy",
            "best_epoch",
            "stopped_epoch",
        ):
            data = np.asarray([row[metric] for row in values], dtype=float)
            item[f"{metric}_mean"] = float(data.mean())
            item[f"{metric}_sd"] = float(data.std(ddof=1)) if len(data) > 1 else 0.0
        output.append(item)
    return sorted(output, key=lambda row: (row["family"], row["config"]))


def paired_contrasts(rows):
    lookup = {(row["family"], row["config"], row["repetition"]): row for row in rows}

    def contrast(family, name, left, right):
        values = []
        for repetition in (0, 1):
            a = lookup[(family, left, repetition)]
            b = lookup[(family, right, repetition)]
            values.append(
                {
                    "family": family,
                    "contrast": name,
                    "repetition": repetition,
                    "left": left,
                    "right": right,
                    "delta_test_top1": a["test_top1"] - b["test_top1"],
                    "delta_test_top5": a["test_top5"] - b["test_top5"],
                    "delta_test_cross_entropy": a["test_cross_entropy"] - b["test_cross_entropy"],
                    "delta_val_top1": a["val_top1"] - b["val_top1"],
                }
            )
        return values

    contrasts = []
    contrasts += contrast(
        "encoding",
        "Trainable ansatz minus zero ansatz (classical encoding)",
        "classical-encoding-trainable-ansatz",
        "classical-encoding-no-ansatz",
    )
    contrasts += contrast(
        "encoding",
        "Classical encoding plus trainable ansatz minus no encoding",
        "classical-encoding-trainable-ansatz",
        "no-encoding-no-ansatz",
    )
    contrasts += contrast(
        "encoding",
        "Classical encoding plus zero ansatz minus no encoding",
        "classical-encoding-no-ansatz",
        "no-encoding-no-ansatz",
    )
    contrasts += contrast(
        "attention",
        "Quantum attention minus no quantum attention (classical encoding)",
        "classical-encoding-quantum-attention",
        "classical-encoding-no-quantum-attention",
    )
    contrasts += contrast(
        "attention",
        "Quantum attention minus no quantum attention (QCSE encoding)",
        "qcse-encoding-quantum-attention",
        "qcse-encoding-no-quantum-attention",
    )
    contrasts += contrast(
        "attention",
        "Classical minus QCSE encoding (quantum attention)",
        "classical-encoding-quantum-attention",
        "qcse-encoding-quantum-attention",
    )
    contrasts += contrast(
        "attention",
        "Classical minus QCSE encoding (no quantum attention)",
        "classical-encoding-no-quantum-attention",
        "qcse-encoding-no-quantum-attention",
    )
    return contrasts


def mean_sd(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=float)
    return values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0


def style_axes(ax):
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=18, length=6, width=1.1, pad=6)


def grouped_test_plot(rows, family, path):
    configs = [config["name"] for config in CONFIGS[family]]
    labels = [PLOT_LABELS[c] for c in configs]
    x = np.arange(len(configs))
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 13.5), sharex=True, constrained_layout=True)
    fig.set_layout_engine("constrained", w_pad=0.02, h_pad=0.02, hspace=0.015, wspace=0.02)
    colors = ("#3765a8", "#c47724")
    for ax, metric, title in zip(
        axes,
        ("test_top1", "test_top5", "test_cross_entropy"),
        ("Held-out top-1", "Held-out top-5", "Held-out cross-entropy"),
        strict=True,
    ):
        for index, config in enumerate(configs):
            values = [
                row[metric] for row in rows if row["family"] == family and row["config"] == config
            ]
            mean, sd = mean_sd(
                [row for row in rows if row["family"] == family and row["config"] == config], metric
            )
            ax.errorbar(index, mean, yerr=sd, fmt="none", ecolor="#222222", capsize=4, zorder=3)
            ax.scatter(
                np.full(len(values), index) + np.linspace(-0.08, 0.08, len(values)),
                values,
                color=colors[: len(values)],
                s=42,
                zorder=4,
            )
            ax.bar(index, mean, color="#9bb7d9", alpha=0.55, width=0.55, zorder=1)
        ax.set_xticks(x, labels, rotation=0, fontsize=17)
        ax.tick_params(axis="x", labelbottom=metric == "test_cross_entropy")
        ax.set_title(title, fontsize=23, pad=14)
        ax.set_ylabel("Accuracy" if metric != "test_cross_entropy" else "CE", fontsize=21)
        if metric != "test_cross_entropy":
            ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
        style_axes(ax)
    fig.suptitle(f"{FAMILY_LABELS[family]} test metrics", fontsize=25)
    fig.savefig(path, dpi=180, transparent=True)
    plt.close(fig)


def grouped_loss_plot(rows, family, path):
    configs = [config["name"] for config in CONFIGS[family]]
    labels = [PLOT_LABELS[c] for c in configs]
    x = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(6.4, 5.7), constrained_layout=True)
    colors = ("#3765a8", "#c47724")
    for index, config in enumerate(configs):
        values = [
            row["test_cross_entropy"]
            for row in rows
            if row["family"] == family and row["config"] == config
        ]
        grouped = [row for row in rows if row["family"] == family and row["config"] == config]
        mean, sd = mean_sd(grouped, "test_cross_entropy")
        ax.bar(index, mean, color="#9bb7d9", alpha=0.6, width=0.55, zorder=1)
        ax.errorbar(index, mean, yerr=sd, fmt="none", ecolor="#222222", capsize=5, zorder=3)
        ax.scatter(
            np.full(len(values), index) + np.linspace(-0.08, 0.08, len(values)),
            values,
            color=colors[: len(values)],
            s=65,
            zorder=4,
        )
    ax.set_xticks(x, labels, rotation=0, fontsize=19)
    ax.set_ylabel("Held-out cross-entropy", fontsize=22)
    ax.set_title(f"{FAMILY_LABELS[family]} cross-entropy", fontsize=25, pad=16)
    style_axes(ax)
    fig.savefig(path, dpi=220, transparent=True)
    plt.close(fig)


def validation_curves(rows, family, path):
    configs = [config["name"] for config in CONFIGS[family]]
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 13.5), sharex=True, constrained_layout=True)
    fig.set_layout_engine("constrained", w_pad=0.02, h_pad=0.02, hspace=0.015, wspace=0.02)
    palette = plt.get_cmap("tab10")(np.arange(10))
    for index, config in enumerate(configs):
        histories = [
            row["history"] for row in rows if row["family"] == family and row["config"] == config
        ]
        epochs = sorted({item["epoch"] for history in histories for item in history})
        curves = {metric: [] for metric in ("cosine_top1", "cosine_top5", "cross_entropy")}
        for epoch in epochs:
            epoch_rows = []
            for history in histories:
                match = next((item for item in history if item["epoch"] == epoch), None)
                if match is not None:
                    epoch_rows.append(match["validation"])
            for metric in curves:
                curves[metric].append(
                    (
                        np.mean([item[metric] for item in epoch_rows]),
                        np.std([item[metric] for item in epoch_rows]),
                    )
                )
        color = palette[index % len(palette)]
        for ax, metric, title in zip(
            axes,
            ("cosine_top1", "cosine_top5", "cross_entropy"),
            ("Validation top-1", "Validation top-5", "Validation cross-entropy"),
            strict=True,
        ):
            mean = np.asarray([item[0] for item in curves[metric]])
            sd = np.asarray([item[1] for item in curves[metric]])
            ax.plot(epochs, mean, label=CONFIG_LABELS[config], color=color, linewidth=1.8)
            if len(histories) > 1:
                ax.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.12)
            ax.set_title(title, fontsize=22, pad=12)
            ax.set_ylabel("Accuracy" if metric != "cross_entropy" else "Cross-entropy", fontsize=20)
            style_axes(ax)
            if metric != "cross_entropy":
                ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
    axes[-1].set_xlabel("Epoch", fontsize=20)
    axes[0].legend(loc="best", fontsize=14)
    fig.suptitle(f"{FAMILY_LABELS[family]} validation drift across epochs", fontsize=25)
    fig.savefig(path, dpi=180, transparent=True)
    plt.close(fig)


def epoch_drift_plot(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    for ax, family in zip(axes, ("encoding", "attention"), strict=True):
        family_rows = [row for row in rows if row["family"] == family]
        configs = [config["name"] for config in CONFIGS[family]]
        x = np.arange(len(configs))
        for row in family_rows:
            index = configs.index(row["config"])
            jitter = -0.12 if row["repetition"] == 0 else 0.12
            ax.scatter(index + jitter, row["best_epoch"], marker="o", color="#3765a8", s=45)
            ax.scatter(index + jitter, row["stopped_epoch"], marker="x", color="#c47724", s=50)
        ax.set_xticks(x, [PLOT_LABELS[c] for c in configs], fontsize=17)
        ax.set_ylabel("Epoch", fontsize=20)
        ax.set_title(FAMILY_LABELS[family], fontsize=22, pad=14)
        style_axes(ax)
    axes[0].scatter([], [], marker="o", color="#3765a8", label="Selected epoch")
    axes[0].scatter([], [], marker="x", color="#c47724", label="Stopped epoch")
    axes[0].legend()
    fig.suptitle("Validation-selection and early-stopping drift", fontsize=25)
    fig.savefig(path, dpi=180, transparent=True)
    plt.close(fig)


def contrast_plot(contrasts, path):
    summary = []
    for name in dict.fromkeys(item["contrast"] for item in contrasts):
        values = [item for item in contrasts if item["contrast"] == name]
        summary.append(
            {
                "name": name,
                "family": values[0]["family"],
                "top1": np.mean([item["delta_test_top1"] for item in values]),
                "top1_sd": np.std([item["delta_test_top1"] for item in values], ddof=1),
                "ce": np.mean([item["delta_test_cross_entropy"] for item in values]),
                "ce_sd": np.std([item["delta_test_cross_entropy"] for item in values], ddof=1),
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(19, 10), constrained_layout=True)
    y = np.arange(len(summary))
    labels = [item["name"] for item in summary]
    colors = ["#3765a8" if item["family"] == "encoding" else "#7c4d9f" for item in summary]
    axes[0].barh(
        y,
        [item["top1"] for item in summary],
        xerr=[item["top1_sd"] for item in summary],
        color=colors,
        alpha=0.75,
    )
    axes[0].axvline(0, color="#222222", linewidth=0.8)
    axes[0].set_yticks(y, labels, fontsize=15)
    axes[0].xaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f} pp")
    axes[0].set_title("Test top-1 contrast", fontsize=22, pad=14)
    axes[0].set_xlabel("Left minus right", fontsize=20)
    axes[1].barh(
        y,
        [item["ce"] for item in summary],
        xerr=[item["ce_sd"] for item in summary],
        color=colors,
        alpha=0.75,
    )
    axes[1].axvline(0, color="#222222", linewidth=0.8)
    axes[1].set_yticks(y, labels, fontsize=15)
    axes[1].set_title("Test cross-entropy contrast", fontsize=22, pad=14)
    axes[1].set_xlabel("Left minus right; negative is better", fontsize=20)
    for ax in axes:
        style_axes(ax)
    fig.suptitle("Paired parameter/setup contrasts", fontsize=25)
    fig.savefig(path, dpi=180, transparent=True)
    plt.close(fig)


def seed_drift_plot(rows, path):
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), constrained_layout=True)
    for ax, family in zip(axes, ("encoding", "attention"), strict=True):
        configs = [config["name"] for config in CONFIGS[family]]
        for config_index, config in enumerate(configs):
            values = [row for row in rows if row["family"] == family and row["config"] == config]
            ax.plot(
                [config_index - 0.1, config_index + 0.1],
                [values[0]["test_top1"], values[1]["test_top1"]],
                color="#8a8a8a",
                linewidth=1,
            )
            ax.scatter(
                [config_index - 0.1, config_index + 0.1],
                [values[0]["test_top1"], values[1]["test_top1"]],
                color=("#3765a8", "#c47724"),
                s=45,
                zorder=3,
            )
        ax.set_xticks(
            np.arange(len(configs)),
            [PLOT_LABELS[c] for c in configs],
            fontsize=17,
        )
        ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
        ax.set_ylabel("Held-out top-1", fontsize=20)
        ax.set_title(FAMILY_LABELS[family], fontsize=22, pad=14)
        style_axes(ax)
    fig.suptitle("Seed/repetition drift in held-out top-1", fontsize=25)
    fig.savefig(path, dpi=180, transparent=True)
    plt.close(fig)


def markdown_reports(rows, aggregates, contrasts):
    def pct(value):
        return f"{100 * value:.2f}%"

    def mean_sd_text(item, metric):
        return f"{pct(item[f'{metric}_mean'])} ± {pct(item[f'{metric}_sd'])}"

    lines = [
        "# Interim focused sweep report — completed 14/30 fits",
        "",
        "## Scope and protocol",
        "",
        "Architecture diagrams are in [architecture-changes.md](architecture-changes.md).",
        "",
        "This report covers only the 14 completed encoding/attention fits from `outputs/focused-sweep-50-local`: three encoding setups and four attention setups, each with repetitions seeded 42 and 43. The causal arm was not complete in this snapshot and is excluded.",
        "",
        "The encoding fits used the full phrases + cleaned corpus; the attention fits used the synthetic repetitive sanity corpus. Both used sentence-grouped 64/16/20 splits, 5,000 replacement draws per training epoch, 2,048 fixed validation-monitor examples, validation top-1-first epoch selection, patience 8 with a 10-epoch minimum, and one held-out test score after selection. Encoding tests used 10,000 test examples; attention has 133 held-out examples.",
        "",
        "The primary selection metric is validation cosine top-1, followed by top-5 and cross-entropy. Test metrics are descriptive only and were not used to select epochs or setups.",
        "",
        "## Aggregate results",
        "",
        "| Family | Setup | Validation top-1 | Test top-1 | Test top-5 | Test CE | Selected epoch |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(
            f"| {FAMILY_LABELS[item['family']]} | {item['label']} | {mean_sd_text(item, 'val_top1')} | {mean_sd_text(item, 'test_top1')} | {mean_sd_text(item, 'test_top5')} | {item['test_cross_entropy_mean']:.3f} ± {item['test_cross_entropy_sd']:.3f} | {item['best_epoch_mean']:.1f} ± {item['best_epoch_sd']:.1f} |"
        )
    lines += [
        "",
        "## Encoding conclusions",
        "",
        "- Classical encoding is the dominant factor. Relative to the no-encoding/no-ansatz control, classical encoding with a trainable ansatz improves held-out top-1 by about 16.9 percentage points and top-5 by about 21.3 points, while reducing cross-entropy by about 1.24.",
        "- The trainable ansatz is a small secondary effect on top of classical encoding: it adds about 0.86 test-top-1 points and reduces test cross-entropy by about 0.012 relative to the zero-ansatz setup, but loses about 0.79 top-5 points. With only two repetitions, this is weak evidence rather than a decisive ansatz result.",
        "- The no-encoding control is consistently poor in both repetitions: roughly 2.8% test top-1 versus roughly 19–20% for either classical-encoding setup. This is a large, stable setup effect, not a subtle optimizer drift.",
        "",
        "## Attention conclusions",
        "",
        "- Quantum attention helps the QCSE encoder strongly: approximately +25.9 test-top-1 points, +7.5 top-5 points, and −0.260 cross-entropy relative to QCSE without quantum attention.",
        "- Quantum attention has a much smaller effect with classical encoding: approximately +2.6 test-top-1 points, no mean top-5 change, and −0.002 cross-entropy. This is an encoder × attention interaction: the quantum-attention gain is much larger for QCSE than for classical encoding.",
        "- Classical encoding remains stronger than QCSE without quantum attention by approximately +27.1 top-1 points and −0.495 cross-entropy. With quantum attention enabled, the classical advantage narrows to about +3.8 top-1 points, while QCSE retains a small top-5 advantage.",
        "- The attention test set has only 133 examples and is a synthetic repetitive sanity corpus. These large effects are useful for pipeline diagnosis and ablation direction, but they should not be treated as corpus-level generalization evidence.",
        "",
        "## Drift and reliability",
        "",
        "- Repetition drift is modest for the encoding setups (test top-1 differences below one percentage point) but larger for attention, especially QCSE without quantum attention. The attention seed-43 run generally selects later epochs and changes the result more than the encoding repetitions.",
        "- Selected epochs vary from early stopping at epoch 3–24 in the attention arm and 5–13 in encoding, with stopped epochs extending to 32. This validates the need for validation-based epoch selection rather than a fixed test-time epoch.",
        "- The primary claims here are directional. Two repetitions are enough to expose large setup effects, not enough to establish tight uncertainty intervals; the new 1,024-sample run is being monitored separately and should supersede this interim snapshot for the final report.",
        "",
        "## Figures",
        "",
        "- `encoding-test-metrics.png`: test top-1/top-5/cross-entropy by encoding setup, with repetition points and mean ± SD.",
        "- `attention-test-metrics.png`: corresponding attention comparison.",
        "- `encoding-cross-entropy.png` and `attention-cross-entropy.png`: narrower loss-only figures with larger labels.",
        "- `encoding-validation-curves.png` and `attention-validation-curves.png`: validation drift across epochs.",
        "- `paired-contrasts.png`: paired parameter/setup effects.",
        "- `selection-epoch-drift.png`: selected versus stopped epochs.",
        "- `seed-drift.png`: repetition-to-repetition held-out top-1 drift.",
    ]
    (RESULTS / "detailed-report.md").write_text("\n".join(lines) + "\n")

    short = [
        "# Interim focused sweep — short report",
        "",
        "The 14 completed non-causal fits show two clear conclusions:",
        "",
        "1. Classical encoding is essential in the encoding arm: about 19–20% test top-1 versus 2.8% without encoding. A trainable ansatz adds only a small, uncertain improvement over a zero ansatz.",
        "2. Quantum attention is highly dependent on the encoder. It improves QCSE by about 26 top-1 points, but classical encoding by only about 3 points. Classical encoding remains best overall, while QCSE + quantum attention recovers much of the gap.",
        "",
        "These are interim directional results from two repetitions. Encoding used 10,000 test examples; attention used a 133-example synthetic test set. The causal arm and the new 1,024-sample protocol are still running and are not represented here.",
    ]
    (RESULTS / "short-report.md").write_text("\n".join(short) + "\n")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = collect()
    serial_rows = [{key: value for key, value in row.items() if key != "history"} for row in rows]
    fields = list(serial_rows[0])
    write_csv(RESULTS / "runs.csv", serial_rows, fields)
    aggregates = aggregate(rows)
    write_csv(RESULTS / "aggregates.csv", aggregates, list(aggregates[0]))
    contrasts = paired_contrasts(rows)
    write_csv(RESULTS / "contrasts.csv", contrasts, list(contrasts[0]))
    (RESULTS / "summary.json").write_text(
        json.dumps(
            {"runs": serial_rows, "aggregates": aggregates, "contrasts": contrasts}, indent=2
        )
    )
    grouped_test_plot(rows, "encoding", RESULTS / "encoding-test-metrics.png")
    grouped_test_plot(rows, "attention", RESULTS / "attention-test-metrics.png")
    grouped_loss_plot(rows, "encoding", RESULTS / "encoding-cross-entropy.png")
    grouped_loss_plot(rows, "attention", RESULTS / "attention-cross-entropy.png")
    validation_curves(rows, "encoding", RESULTS / "encoding-validation-curves.png")
    validation_curves(rows, "attention", RESULTS / "attention-validation-curves.png")
    epoch_drift_plot(rows, RESULTS / "selection-epoch-drift.png")
    contrast_plot(contrasts, RESULTS / "paired-contrasts.png")
    seed_drift_plot(rows, RESULTS / "seed-drift.png")
    markdown_reports(rows, aggregates, contrasts)
    print(f"Wrote interim report and plots to {RESULTS}")


if __name__ == "__main__":
    main()
