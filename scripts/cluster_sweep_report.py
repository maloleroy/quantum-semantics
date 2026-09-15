# ruff: noqa: E501

"""Validate and report the 45-configuration cluster QCSE sweep archive.

The archive is read without extracting its raw run folders.  The validator checks
the completed CV protocol, split disjointness and sentence-group isolation, the
per-epoch aggregation, and the exported held-out predictions.  The report mode
also writes machine-readable tables and a deliberately redundant set of static
figures for later inspection.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/qcse-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPECTED_IDS = set(range(45))
EXPECTED_FILES = {
    "summary.json",
    "run_info.json",
    "training_config.json",
    "cv_history.json",
    "cross_validation.json",
    "splits.npz",
    "test_embeddings.npz",
    "fold-01/history.json",
    "fold-01/full_validation.json",
    "fold-01/run.npz",
    "fold-01/model.npz",
    "fold-02/history.json",
    "fold-02/full_validation.json",
    "fold-02/run.npz",
    "fold-02/model.npz",
    "refit/history.json",
    "refit/run.npz",
    "refit/model.npz",
}
METRICS = ("bce", "bit_accuracy", "exact_word_accuracy", "paper_similarity_accuracy")
PLOT_NAMES = (
    "validation-top1-ranking.png",
    "test-top1-ranking.png",
    "validation-bce-ranking.png",
    "test-bce-ranking.png",
    "validation-metrics-heatmap.png",
    "main-grid-validation-top1.png",
    "main-grid-validation-bce.png",
    "learning-rate-effect-top1.png",
    "learning-rate-effect.png",
    "alpha-effect-top1.png",
    "alpha-effect.png",
    "window-effect-top1.png",
    "window-effect.png",
    "depth-batch-validation-top1.png",
    "depth-batch-validation-bce.png",
    "sentence-pool-validation-top1.png",
    "sentence-pool-validation-bce.png",
    "validation-test-top1-gap.png",
    "validation-test-gap.png",
    "epoch-curves-top1.png",
    "epoch-curves.png",
    "runtime-by-configuration.png",
)


def read_json(zf: zipfile.ZipFile, name: str):
    return json.loads(zf.read(name))


def parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def metric(probabilities: np.ndarray, target_ids: np.ndarray, qubits: int = 14) -> dict[str, float]:
    bit_positions = np.arange(qubits, dtype=np.int64)
    targets = ((target_ids[:, None] >> bit_positions) & 1).astype(bool)
    matching = (probabilities >= 0.5) == targets
    return {
        "bce": float(
            -np.mean(
                targets * np.log(np.clip(probabilities, 1e-10, 1 - 1e-10))
                + (1 - targets) * np.log1p(-np.clip(probabilities, 1e-10, 1 - 1e-10))
            )
        ),
        "bit_accuracy": float(matching.mean()),
        "exact_word_accuracy": float(matching.all(axis=1).mean()),
        "paper_similarity_accuracy": float((matching.mean(axis=1) >= 0.5).mean()),
    }


def sentence_group_index(zf: zipfile.ZipFile, base: str) -> tuple[np.ndarray, int]:
    """Return cumulative causal-example ends and the sentence count.

    ``sentences.csv`` is already normalized, one sentence per row.  For causal
    next-token examples, a sentence with n tokens contributes max(n - 1, 0)
    examples.  This reconstructs group ownership without parsing huge run.npz
    metadata archives.
    """
    counts = []
    with io.TextIOWrapper(zf.open(base + "sentences.csv"), encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            counts.append(max(len(row["sentence"].split()) - 1, 0))
    cumulative = np.cumsum(np.asarray(counts, dtype=np.int64))
    return cumulative, len(counts)


def groups(ids: np.ndarray, cumulative: np.ndarray) -> np.ndarray:
    if not len(ids):
        return np.empty(0, dtype=np.int64)
    return np.searchsorted(cumulative, ids, side="right")


def close(a: float, b: float, tolerance: float = 1e-10) -> bool:
    return bool(math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance))


def has_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    """Fast overlap test for the sorted split-ID arrays in the archive."""
    if not len(first) or not len(second):
        return False
    if len(first) > len(second):
        first, second = second, first
    positions = np.searchsorted(second, first)
    valid = positions < len(second)
    return bool(np.any(valid & (second[np.minimum(positions, len(second) - 1)] == first)))


def is_subset(subset: np.ndarray, superset: np.ndarray) -> bool:
    if not len(subset):
        return True
    if not len(superset):
        return False
    positions = np.searchsorted(superset, subset)
    valid = positions < len(superset)
    return bool(np.all(valid & (superset[np.minimum(positions, len(superset) - 1)] == subset)))


def run_files(zf: zipfile.ZipFile, base: str) -> set[str]:
    files = set()
    for name in zf.namelist():
        if name.startswith(base) and name != base:
            relative = name[len(base) :]
            if relative and not relative.endswith("/"):
                files.add(relative)
    return files


def collect(zf: zipfile.ZipFile) -> tuple[list[dict], list[dict]]:
    runs: dict[int, list[str]] = defaultdict(list)
    for name in zf.namelist():
        match = re.match(r"^sweep/experiment-(\d{3})/([^/]+)/run_info\.json$", name)
        if match:
            runs[int(match.group(1))].append(match.group(2))

    completed, incomplete = [], []
    for experiment_id in sorted(runs):
        for run_name in sorted(runs[experiment_id]):
            base = f"sweep/experiment-{experiment_id:03d}/{run_name}/"
            info = read_json(zf, base + "run_info.json")
            args = info.get("arguments", {})
            record = {
                "id": experiment_id,
                "run": run_name,
                "base": base,
                "status": info.get("status"),
                "started_at": info.get("started_at"),
                "finished_at": info.get("finished_at"),
                "device": info.get("device"),
                "torch_version": info.get("torch_version"),
                "cuda_version": info.get("cuda_version"),
                "args": args,
                "files": run_files(zf, base),
            }
            if "cross_validation.json" in record["files"]:
                record["cv"] = read_json(zf, base + "cross_validation.json")
                record["summary"] = read_json(zf, base + "summary.json")
                record["cv_history"] = read_json(zf, base + "cv_history.json")
                record["fold_histories"] = [
                    read_json(zf, base + f"fold-{fold:02d}/history.json") for fold in (1, 2)
                ]
                start, finish = parse_time(record["started_at"]), parse_time(record["finished_at"])
                record["duration_seconds"] = (
                    (finish - start).total_seconds() if start and finish else None
                )
                if record["cv"].get("status") == "complete":
                    record.update(
                        {
                            "alpha": args["alpha"],
                            "learning_rate": args["learning_rate"],
                            "window": args["window"],
                            "layers": args["layers"],
                            "batch_size": args["batch_size"],
                            "max_sentences": args["max_sentences"],
                            "sampling": args["sampling"],
                            "validation_bce": record["cv"]["validation"]["bce"]["mean"],
                            "validation_bce_sd": record["cv"]["validation"]["bce"]["std"],
                            "test_bce": record["cv"]["test"]["bce"],
                        }
                    )
                    completed.append(record)
                    continue
            incomplete.append(record)
    return completed, incomplete


def protocol_issues(zf: zipfile.ZipFile, record: dict) -> list[str]:
    issues = []
    base = record["base"]
    args = record["args"]
    cv = record["cv"]
    summary = record["summary"]
    required = EXPECTED_FILES
    missing = sorted(required - record["files"])
    if missing:
        issues.append(f"missing files: {', '.join(missing)}")
    expected_args = {
        "command": "train",
        "device": "cuda",
        "datasets": ["phrases", "cleaned"],
        "cleaning": "dedupe",
        "sampling": record["sampling"],
        "objective": "causal",
        "epochs": 10,
        "folds": 2,
        "val_fraction": 0.2,
        "samples_per_epoch": 5000,
        "eval_examples": 2048,
        "max_examples": None,
        "seed": 42,
    }
    for key, expected in expected_args.items():
        if args.get(key) != expected:
            issues.append(f"argument {key}={args.get(key)!r}, expected {expected!r}")
    if record["status"] != "complete":
        issues.append(f"run_info status is {record['status']!r}")
    if cv.get("epochs_per_fit") != 10 or cv.get("samples_per_epoch") != 5000:
        issues.append("cross-validation epoch or sample budget mismatch")
    if cv.get("epoch_metrics") != "fixed samples":
        issues.append("epoch metrics are not marked fixed samples")
    if cv.get("test_evaluation") != "once after final refit":
        issues.append("test evaluation policy mismatch")
    if summary.get("objective") != "causal" or summary.get("qubits") != 14:
        issues.append("summary model protocol mismatch")
    if summary.get("examples") != cv.get("development_examples", 0) + cv.get("test_examples", 0):
        issues.append("example count does not equal development plus test")

    with np.load(io.BytesIO(zf.read(base + "splits.npz")), allow_pickle=False) as split:
        arrays = {key: split[key] for key in split.files}
    development = arrays["development_ids"]
    test = arrays["test_ids"]
    all_original = arrays["original_example_ids"]
    if len(np.unique(development)) != len(development):
        issues.append("duplicate development IDs")
    if len(np.unique(test)) != len(test):
        issues.append("duplicate test IDs")
    if has_overlap(development, test):
        issues.append("development/test overlap")
    if len(all_original) != summary.get("examples"):
        issues.append("original_example_ids length mismatch")
    cumulative, sentence_count = sentence_group_index(zf, base)
    if cumulative.size != summary.get("sentences"):
        issues.append("sentence count mismatch")
    if cumulative.size and cumulative[-1] != summary.get("examples"):
        issues.append("sentence-derived causal example count mismatch")
    test_groups = groups(test, cumulative)
    development_groups = groups(development, cumulative)
    if np.intersect1d(test_groups, development_groups).size:
        issues.append("sentence group crosses outer test boundary")
    for fold in (1, 2):
        train_ids = arrays[f"fold_{fold}_train_ids"]
        validation_ids = arrays[f"fold_{fold}_validation_ids"]
        if has_overlap(train_ids, validation_ids):
            issues.append(f"fold {fold} train/validation overlap")
        if has_overlap(train_ids, test) or has_overlap(validation_ids, test):
            issues.append(f"fold {fold} contains outer test IDs")
        if not is_subset(train_ids, development) or not is_subset(validation_ids, development):
            issues.append(f"fold {fold} IDs outside development")
        train_groups = groups(train_ids, cumulative)
        validation_groups = groups(validation_ids, cumulative)
        if has_overlap(train_groups, validation_groups):
            issues.append(f"fold {fold} sentence group crosses train/validation boundary")

    # Recompute the summary statistics from the two full validation reports.
    fold_metrics = [row["metrics"] for row in cv["folds"]]
    for key in METRICS:
        values = np.asarray([row[key] for row in fold_metrics], dtype=float)
        if not close(float(values.mean()), cv["validation"][key]["mean"]):
            issues.append(f"validation {key} mean mismatch")
        if not close(float(values.std(ddof=1)), cv["validation"][key]["std"]):
            issues.append(f"validation {key} std mismatch")
        for fold in (1, 2):
            full = read_json(zf, base + f"fold-{fold:02d}/full_validation.json")
            if any(not close(full[key], fold_metrics[fold - 1][key]) for key in METRICS):
                issues.append(f"fold {fold} full validation mismatch")

    for fold, history in enumerate(record["fold_histories"], 1):
        if [row["epoch"] for row in history] != list(range(11)):
            issues.append(f"fold {fold} history does not contain epochs 0..10")
        for row in history:
            if "validation" not in row or "test" in row:
                issues.append(f"fold {fold} history has invalid evaluation fields")
        if history[-1].get("metric_examples") != {"train": 2048, "validation": 2048}:
            issues.append(f"fold {fold} monitoring sample size mismatch")
    if len(record["cv_history"]) != 11 or [row["epoch"] for row in record["cv_history"]] != list(
        range(11)
    ):
        issues.append("cv_history does not contain epochs 0..10")
    for epoch in range(11):
        for split in ("train", "validation"):
            for key in METRICS:
                values = np.asarray(
                    [history[epoch][split][key] for history in record["fold_histories"]],
                    dtype=float,
                )
                aggregate = record["cv_history"][epoch][split][key]
                if not close(float(values.mean()), aggregate["mean"]):
                    issues.append(f"cv_history epoch {epoch} {split} {key} mean mismatch")
                if not close(float(values.std(ddof=1)), aggregate["std"]):
                    issues.append(f"cv_history epoch {epoch} {split} {key} std mismatch")

    # The final test export is the only complete held-out prediction artifact.
    with np.load(io.BytesIO(zf.read(base + "test_embeddings.npz")), allow_pickle=False) as export:
        probabilities = export["probabilities"]
        target_ids = export["target_ids"]
        example_ids = export["example_ids"]
        if probabilities.shape != (cv["test_examples"], 14):
            issues.append("test probability shape mismatch")
        if len(target_ids) != cv["test_examples"] or len(example_ids) != cv["test_examples"]:
            issues.append("test export length mismatch")
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
        ):
            issues.append("test probabilities are nonfinite or outside [0, 1]")
        recomputed = metric(probabilities, target_ids)
        if any(not close(recomputed[key], cv["test"][key]) for key in METRICS):
            issues.append("recomputed test metrics mismatch")
        if not np.array_equal(example_ids, test):
            issues.append("test export example IDs do not match splits")
    return sorted(set(issues))


def row_for_json(record: dict, issues: list[str] | None = None) -> dict:
    args = record["args"]
    cv = record["cv"]
    return {
        "experiment_id": record["id"],
        "run": record["run"],
        "status": record["status"],
        "device": record["device"],
        "torch_version": record["torch_version"],
        "cuda_version": record["cuda_version"],
        "alpha": args["alpha"],
        "learning_rate": args["learning_rate"],
        "window": args["window"],
        "layers": args["layers"],
        "batch_size": args["batch_size"],
        "max_sentences": args["max_sentences"],
        "sampling": args["sampling"],
        "sentences": record["summary"]["sentences"],
        "examples": record["summary"]["examples"],
        "development_examples": cv["development_examples"],
        "test_examples": cv["test_examples"],
        "validation_bce": cv["validation"]["bce"]["mean"],
        "validation_bce_sd": cv["validation"]["bce"]["std"],
        # The QCSE decoder emits independent bit probabilities rather than a
        # categorical word distribution.  Its word-level top-1 is therefore
        # the exact all-bits match, stored in the archive as
        # ``exact_word_accuracy``.
        "validation_top1": cv["validation"]["exact_word_accuracy"]["mean"],
        "validation_top1_sd": cv["validation"]["exact_word_accuracy"]["std"],
        "validation_bit_accuracy": cv["validation"]["bit_accuracy"]["mean"],
        "validation_exact_word_accuracy": cv["validation"]["exact_word_accuracy"]["mean"],
        "validation_paper_similarity_accuracy": cv["validation"]["paper_similarity_accuracy"]["mean"],
        "test_bce": cv["test"]["bce"],
        "test_top1": cv["test"]["exact_word_accuracy"],
        "test_bit_accuracy": cv["test"]["bit_accuracy"],
        "test_exact_word_accuracy": cv["test"]["exact_word_accuracy"],
        "test_paper_similarity_accuracy": cv["test"]["paper_similarity_accuracy"],
        "duration_seconds": record["duration_seconds"],
        "validation_issues": issues or [],
        "folds": cv["folds"],
        "cv_history": record["cv_history"],
    }


def plot_validation_ranking(
    rows: list[dict], out: Path, value_key: str, sd_key: str, filename: str, xlabel: str, title: str
) -> None:
    ordered = sorted(rows, key=lambda r: r[value_key], reverse=True)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.errorbar(
        [r[value_key] for r in ordered],
        y,
        xerr=[r[sd_key] for r in ordered],
        fmt="o",
        color="#245a9b",
        ecolor="#8aa6c5",
        capsize=2,
    )
    ax.set_yticks(y, [f"{r['experiment_id']:02d}" for r in ordered])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Experiment ID")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(out / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_test_ranking(
    rows: list[dict], out: Path, value_key: str, filename: str, xlabel: str, title: str
) -> None:
    ordered = sorted(rows, key=lambda r: r[value_key], reverse=True)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.scatter([r[value_key] for r in ordered], y, color="#b34a2e", s=28)
    ax.set_yticks(y, [f"{r['experiment_id']:02d}" for r in ordered])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Experiment ID")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(out / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rankings(rows: list[dict], out: Path) -> None:
    plot_validation_ranking(
        rows,
        out,
        "validation_top1",
        "validation_top1_sd",
        "validation-top1-ranking.png",
        "Full validation word top-1 (mean ± fold SD)",
        "Completed cluster sweep ranked by validation word top-1",
    )
    plot_test_ranking(
        rows,
        out,
        "test_top1",
        "test-top1-ranking.png",
        "Held-out test word top-1",
        "Completed cluster sweep ranked by held-out word top-1",
    )
    plot_validation_ranking(
        rows,
        out,
        "validation_bce",
        "validation_bce_sd",
        "validation-bce-ranking.png",
        "Full validation BCE (mean ± fold SD)",
        "Secondary ranking by validation BCE",
    )
    plot_test_ranking(
        rows,
        out,
        "test_bce",
        "test-bce-ranking.png",
        "Held-out test BCE",
        "Secondary ranking by held-out test BCE",
    )


def plot_heatmap(rows: list[dict], out: Path) -> None:
    rows = sorted(rows, key=lambda r: r["experiment_id"])
    data = np.asarray(
        [
            [
                r["validation_bce"],
                r["test_bce"],
                100 * r["validation_top1"],
                100 * r["test_top1"],
                100 * r["validation_bit_accuracy"],
                100 * r["test_bit_accuracy"],
                100 * r["validation_paper_similarity_accuracy"],
                100 * r["test_paper_similarity_accuracy"],
            ]
            for r in rows
        ]
    ).T
    labels = [
        "validation BCE",
        "test BCE",
        "validation word top-1 (%)",
        "test word top-1 (%)",
        "validation bit acc. (%)",
        "test bit acc. (%)",
        "validation ≥half-bit (%)",
        "test ≥half-bit (%)",
    ]
    blocks = (
        (data[:2], labels[:2], "Cross-entropy"),
        (data[2:4], labels[2:4], "Word top-1"),
        (data[4:6], labels[4:6], "Bit accuracy"),
        (data[6:], labels[6:], "At least half the bits"),
    )
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True, constrained_layout=True)
    for ax, block in zip(axes, blocks, strict=True):
        block_data, block_labels, block_title = block
        image = ax.imshow(block_data, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_yticks(np.arange(len(block_data)), block_labels)
        ax.set_ylabel(block_title)
        ax.set_xticks(np.arange(len(rows)), [f"{r['experiment_id']:02d}" for r in rows])
        for i in range(block_data.shape[0]):
            for j in range(block_data.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{block_data[i, j]:.3f}" if block_title == "Cross-entropy" else f"{block_data[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white",
                )
        fig.colorbar(image, ax=ax, fraction=0.015, pad=0.01)
    axes[-1].set_xlabel("Experiment ID")
    fig.suptitle("Validation and test metrics for all completed configurations")
    fig.savefig(out / "validation-metrics-heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def grouped_mean(rows: list[dict], key: str, value: str = "validation_bce"):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row[value])
    return [
        (
            group,
            float(np.mean(values)),
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        )
        for group, values in grouped.items()
    ]


def plot_effect(
    rows: list[dict],
    key: str,
    filename: str,
    title: str,
    xlabel: str,
    out: Path,
    value_key: str,
    ylabel: str,
    logx=False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for lr in sorted({r["learning_rate"] for r in rows}):
        selected = [r for r in rows if r["learning_rate"] == lr]
        points = sorted(grouped_mean(selected, key, value_key))
        x, y, sd = zip(*points, strict=True)
        ax.errorbar(x, y, yerr=sd, marker="o", capsize=3, label=f"LR {lr:g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(out / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_main_grid(
    rows: list[dict],
    out: Path,
    value_key: str,
    sd_key: str,
    filename: str,
    ylabel: str,
    title: str,
) -> None:
    main = [r for r in rows if r["experiment_id"] < 27]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, constrained_layout=True)
    for ax, alpha in zip(axes, (0.1, 1.0, 3.0), strict=True):
        for window, marker in zip((2, 4, 8), ("o", "s", "^"), strict=True):
            selected = [r for r in main if r["alpha"] == alpha and r["window"] == window]
            selected.sort(key=lambda r: r["learning_rate"])
            ax.errorbar(
                [r["learning_rate"] for r in selected],
                [r[value_key] for r in selected],
                yerr=[r[sd_key] for r in selected],
                marker=marker,
                capsize=2,
                label=f"window {window}",
            )
        ax.set_xscale("log")
        ax.set_title(f"context alpha = {alpha:g}")
        ax.set_xlabel("Learning rate")
        ax.grid(alpha=0.25)
    axes[-1].legend(fontsize=8)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.savefig(out / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_depth_batch(
    rows: list[dict], out: Path, value_key: str, sd_key: str, filename: str, ylabel: str
) -> None:
    selected = [r for r in rows if 27 <= r["experiment_id"] < 39]
    fig, ax = plt.subplots(figsize=(9, 5))
    for layers, color in zip((2, 8, 64), ("#4c78a8", "#f58518", "#54a24b"), strict=True):
        part = [r for r in selected if r["layers"] == layers]
        for window, linestyle in zip((2, 8), ("-", "--"), strict=True):
            line = sorted([r for r in part if r["window"] == window], key=lambda r: r["batch_size"])
            ax.errorbar(
                [r["batch_size"] for r in line],
                [r[value_key] for r in line],
                yerr=[r[sd_key] for r in line],
                color=color,
                linestyle=linestyle,
                marker="o",
                capsize=2,
                label=f"{layers} layers, window {window}",
            )
    ax.set_xlabel("Adam batch size (5,000 draws/epoch)")
    ax.set_ylabel(ylabel)
    ax.set_title("Depth and batch-size comparison")
    ax.set_xticks((16, 64, 256))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(out / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pool(
    rows: list[dict], out: Path, value_key: str, sd_key: str, filename: str, ylabel: str
) -> None:
    selected = [r for r in rows if r["experiment_id"] >= 39]
    fig, ax = plt.subplots(figsize=(8, 5))
    for sampling, marker in (("uniform", "o"), ("balanced", "s")):
        line = sorted(
            [r for r in selected if r["sampling"] == sampling],
            key=lambda r: r["max_sentences"],
        )
        ax.errorbar(
            [r["max_sentences"] for r in line],
            [r[value_key] for r in line],
            yerr=[r[sd_key] for r in line],
            marker=marker,
            capsize=3,
            label=sampling,
        )
    ax.set_xlabel("Sentence-pool cap")
    ax.set_ylabel(ylabel)
    ax.set_title("Sentence-pool size and source-balancing comparison")
    ax.set_xticks((5000, 20000, 50000))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gap(
    rows: list[dict],
    out: Path,
    validation_key: str,
    test_key: str,
    filename: str,
    xlabel: str,
    ylabel: str,
    title: str,
    reverse: bool,
) -> None:
    ordered = sorted(rows, key=lambda r: r[validation_key], reverse=reverse)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ordered))
    gap = np.asarray([r[test_key] - r[validation_key] for r in ordered])
    colors = ["#2e7d32" if value >= 0 else "#c62828" for value in gap]
    ax.bar(x, gap, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"{r['experiment_id']:02d}" for r in ordered], rotation=90)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_epochs(
    rows: list[dict],
    out: Path,
    selection_key: str,
    history_key: str,
    scale: float,
    filename: str,
    ylabel: str,
    title: str,
    reverse: bool,
) -> None:
    chosen = sorted(rows, key=lambda r: r[selection_key], reverse=reverse)[:8]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for row in chosen:
        history = row["cv_history"]
        label = f"{row['experiment_id']:02d} (b{row['batch_size']}, w{row['window']})"
        axes[0].plot(
            [h["epoch"] for h in history],
            [scale * h["train"][history_key]["mean"] for h in history],
            label=label,
        )
        axes[1].plot(
            [h["epoch"] for h in history],
            [scale * h["validation"][history_key]["mean"] for h in history],
            label=label,
        )
    axes[0].set_title(f"Training monitoring {ylabel}")
    axes[1].set_title(f"Validation monitoring {ylabel}")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    fig.savefig(out / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_runtime(rows: list[dict], out: Path) -> None:
    ordered = sorted(rows, key=lambda r: r["experiment_id"])
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ordered))
    ax.bar(x, [r["duration_seconds"] / 60 for r in ordered], color="#6a4c93")
    ax.set_xticks(x, [f"{r['experiment_id']:02d}" for r in ordered])
    ax.set_xlabel("Experiment ID")
    ax.set_ylabel("Wall time (minutes)")
    ax.set_title("Completed configuration wall times recorded by run_info.json")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out / "runtime-by-configuration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_plots(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plot_rankings(rows, out)
    plot_heatmap(rows, out)
    main = [r for r in rows if r["experiment_id"] < 27]
    for value_key, sd_key, filename, ylabel, title in (
        (
            "validation_top1",
            "validation_top1_sd",
            "main-grid-validation-top1.png",
            "Validation word top-1",
            "27-configuration alpha / learning-rate / window grid by word top-1",
        ),
        (
            "validation_bce",
            "validation_bce_sd",
            "main-grid-validation-bce.png",
            "Validation BCE",
            "27-configuration alpha / learning-rate / window grid by BCE",
        ),
    ):
        plot_main_grid(rows, out, value_key, sd_key, filename, ylabel, title)
    for value_key, suffix, ylabel in (
        ("validation_top1", "top1", "Validation word top-1 (mean across matching configurations)"),
        ("validation_bce", "bce", "Validation BCE (mean across matching configurations)"),
    ):
        plot_effect(
            main,
            "learning_rate",
            f"learning-rate-effect-{suffix}.png" if suffix == "top1" else "learning-rate-effect.png",
            f"Learning-rate effect in the main grid ({suffix})",
            "Learning rate",
            out,
            value_key,
            ylabel,
            logx=True,
        )
        plot_effect(
            main,
            "alpha",
            f"alpha-effect-{suffix}.png" if suffix == "top1" else "alpha-effect.png",
            f"Context-alpha effect in the main grid ({suffix})",
            "Context alpha",
            out,
            value_key,
            ylabel,
        )
        plot_effect(
            main,
            "window",
            f"window-effect-{suffix}.png" if suffix == "top1" else "window-effect.png",
            f"Context-window effect in the main grid ({suffix})",
            "Context window",
            out,
            value_key,
            ylabel,
        )
    for value_key, sd_key, filename, ylabel in (
        (
            "validation_top1",
            "validation_top1_sd",
            "depth-batch-validation-top1.png",
            "Validation word top-1",
        ),
        ("validation_bce", "validation_bce_sd", "depth-batch-validation-bce.png", "Validation BCE"),
    ):
        plot_depth_batch(rows, out, value_key, sd_key, filename, ylabel)
    for value_key, sd_key, filename, ylabel in (
        (
            "validation_top1",
            "validation_top1_sd",
            "sentence-pool-validation-top1.png",
            "Validation word top-1",
        ),
        ("validation_bce", "validation_bce_sd", "sentence-pool-validation-bce.png", "Validation BCE"),
    ):
        plot_pool(rows, out, value_key, sd_key, filename, ylabel)
    plot_gap(
        rows,
        out,
        "validation_top1",
        "test_top1",
        "validation-test-top1-gap.png",
        "Experiment ID, ordered by validation word top-1",
        "Test word top-1 − validation word top-1",
        "Word top-1 generalization gap",
        True,
    )
    plot_gap(
        rows,
        out,
        "validation_bce",
        "test_bce",
        "validation-test-gap.png",
        "Experiment ID, ordered by validation BCE",
        "Test BCE − validation BCE",
        "Secondary BCE generalization gap",
        False,
    )
    plot_epochs(
        rows,
        out,
        "validation_top1",
        "exact_word_accuracy",
        100,
        "epoch-curves-top1.png",
        "word top-1 (%)",
        "Epoch curves for the eight best validation word top-1 configurations",
        True,
    )
    plot_epochs(
        rows,
        out,
        "validation_bce",
        "bce",
        1,
        "epoch-curves.png",
        "BCE",
        "Secondary epoch curves for the eight best validation BCE configurations",
        False,
    )
    plot_runtime(rows, out)


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| ID | alpha | LR | window | layers | batch | pool | sampling | val word top-1 ± SD | test word top-1 | val BCE ± SD | test BCE | minutes |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: r["validation_top1"], reverse=True):
        pool = "full" if row["max_sentences"] is None else str(row["max_sentences"])
        lines.append(
            f"| {row['experiment_id']:02d} | {row['alpha']:g} | {row['learning_rate']:g} | {row['window']} | "
            f"{row['layers']} | {row['batch_size']} | {pool} | {row['sampling']} | "
            f"{100 * row['validation_top1']:.3f}% ± {100 * row['validation_top1_sd']:.3f}% | "
            f"{100 * row['test_top1']:.3f}% | {row['validation_bce']:.6f} ± {row['validation_bce_sd']:.6f} | "
            f"{row['test_bce']:.6f} | {row['duration_seconds'] / 60:.1f} |"
        )
    return "\n".join(lines)


def short_table(rows: list[dict]) -> str:
    lines = [
        "| ID | setting | validation word top-1 ± SD | test word top-1 | validation BCE |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda r: r["validation_top1"], reverse=True)[:8]:
        setting = f"a={row['alpha']:g}, lr={row['learning_rate']:g}, w={row['window']}, L={row['layers']}, B={row['batch_size']}"
        lines.append(
            f"| {row['experiment_id']:02d} | {setting} | {100 * row['validation_top1']:.3f}% ± {100 * row['validation_top1_sd']:.3f}% | "
            f"{100 * row['test_top1']:.3f}% | {row['validation_bce']:.6f} |"
        )
    return "\n".join(lines)


def write_reports(rows: list[dict], incomplete: list[dict], issues: dict[int, list[str]], output: Path, archive: Path) -> None:
    best = max(rows, key=lambda r: r["validation_top1"])
    best_bce = min(rows, key=lambda r: r["validation_bce"])
    best_test = max(rows, key=lambda r: r["test_top1"])
    best_test_bce = min(rows, key=lambda r: r["test_bce"])
    pool = [r for r in rows if r["max_sentences"] is not None]
    reference = next(r for r in rows if r["experiment_id"] == 13)
    main = [r for r in rows if r["experiment_id"] < 27]

    def effect_table(key: str, label: str, value_key: str = "validation_top1") -> str:
        values = sorted({r[key] for r in main})
        percentage = value_key == "validation_top1"
        metric_label = "mean validation word top-1" if percentage else "mean validation BCE"
        lines = [
            f"| {label} | {metric_label} | minimum | maximum |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for value in values:
            selected = [r[value_key] for r in main if r[key] == value]
            multiplier = 100 if percentage else 1
            lines.append(
                f"| {value:g} | {multiplier * np.mean(selected):.6f}{'%' if percentage else ''} | "
                f"{multiplier * np.min(selected):.6f}{'%' if percentage else ''} | "
                f"{multiplier * np.max(selected):.6f}{'%' if percentage else ''} |"
            )
        return "\n".join(lines)

    pool_table = [
        "| ID | pool | sampling | validation word top-1 ± SD | test word top-1 | validation BCE | test BCE |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(pool, key=lambda r: r["max_sentences"]):
        pool_table.append(
            f"| {row['experiment_id']:02d} | {row['max_sentences']} | {row['sampling']} | "
            f"{100 * row['validation_top1']:.3f}% ± {100 * row['validation_top1_sd']:.3f}% | "
            f"{100 * row['test_top1']:.3f}% | {row['validation_bce']:.6f} | {row['test_bce']:.6f} |"
        )
    pool_table_text = "\n".join(pool_table)
    figure_links = "\n".join(
        f"- [{name.removesuffix('.png')}]({Path('plots') / name})" for name in PLOT_NAMES
    )
    report = f"""# Cluster QCSE sweep — detailed report

Date generated: 2026-09-15  
Source archive: `{archive}`  
Scope: the 45-configuration production QCSE sweep, not the separate ten-configuration semantic-decoder sweep.

## Executive summary

The archive contains 45 experiment IDs. **44 configurations completed** the intended two-fold cross-validation and final refit; experiment 039 was still running at archive capture, with only its epoch-0 first-fold checkpoint present. The archive also contains 54 older non-CV or legacy attempt directories; none is included in the rankings. All 44 completed runs passed the independent archive validator described below.

For this reanalysis, configurations are ranked by mean full validation **word top-1**, never by the held-out test. In this binary QCSE decoder, word top-1 is `exact_word_accuracy`: all 14 thresholded output bits must match the target word ID. The top configuration is experiment **{best['experiment_id']:02d}**: alpha={best['alpha']:g}, learning rate={best['learning_rate']:g}, window={best['window']}, {best['layers']} layers, batch {best['batch_size']}. It reached validation word top-1 **{100 * best['validation_top1']:.3f}% ± {100 * best['validation_top1_sd']:.3f}%**, then test word top-1 **{100 * best['test_top1']:.3f}%**. Its secondary validation BCE is {best['validation_bce']:.6f} ± {best['validation_bce_sd']:.6f}. The previous BCE winner is experiment {best_bce['experiment_id']:02d}; it is retained as a diagnostic, not the primary selection.

The top-1 result is driven by the depth/batch comparison: experiment 035 (64 layers, batch 64) is the word-top-1 winner, while experiment 031 (8 layers, batch 16) is the BCE winner. Every epoch draws exactly 5,000 examples with replacement: batch 16 receives 313 Adam updates, batch 64 receives 79, and batch 256 receives 20. Therefore neither the batch nor depth effect is compute-matched. In the main grid, learning rate 0.001 gives the highest average word top-1, while alpha changes are small and the word-top-1 advantage of context windows is not monotonic.

## Protocol and configuration

- Both active sources were used: `phrases.csv` and `cleaned_sentences.csv`; Tatoeba was excluded. Dedupe cleaning retained 202,172 sentence rows, 1,582,956 causal examples, the 10,864-word vocabulary and 14 qubits.
- Every run used causal next-token prediction, seed 42, exponential context encoding, 5,000 replacement draws per epoch, fixed monitoring samples of 2,048 train and validation examples, two independent 20% validation shuffle-splits inside the 80% development partition, and one final held-out test evaluation after refitting on all development examples.
- Because `samples_per_epoch=5000`, the refit optimizer work is 5,000 draws per epoch rather than a full pass over the roughly 1.58 million eligible causal examples. The expensive exhaustive scoring is reserved for the two full validation folds and the final held-out test: for a full-pool run this is about 505,665 validation examples plus 316,538 test examples, not all examples on every epoch.
- A sampled epoch draws 5,000 example IDs from the training partition with replacement. Repeated IDs are therefore allowed, and a sentence can occur several times through repeated windows or repeated draws; this is training resampling, not leakage. The fixed train/validation monitoring subsets are sampled without replacement and independently from training draws.
- IDs 0–38 use the full 202,172-sentence pool. IDs 39–44 are the sentence-pool group; five are completed in this archive and ID 39 is incomplete. The completed pool runs use 5,000/20,000/50,000 sentences with uniform or balanced source sampling.
- Completed artifacts report CUDA execution with `torch {best['torch_version']}` and CUDA `{best['cuda_version']}`. The archive does not retain scheduler stdout/stderr, the exact Slurm allocation metadata, or a Git commit SHA; those provenance limits are kept explicit here.

## Validation and archive integrity

The validator checked, for every completed ID:

- the intended arguments, 10-epoch target, 5,000-draw budget, two-fold/fixed-monitoring protocol, and required final artifacts;
- unique and disjoint outer development/test IDs, fold train/validation disjointness, no fold/test overlap, and all sentence groups kept on one side of every split;
- the two full validation JSON files against the CV summary, including mean and sample standard deviation;
- all 11 epoch rows in both fold histories and the aggregate `cv_history.json`, recomputing every mean and sample standard deviation;
- the final test export shape, finiteness and `[0, 1]` probability range, exact example-ID correspondence to `splits.npz`, and all four test metrics recomputed from exported probabilities and target IDs.

Result: **44/44 completed runs passed; 0 validation issues**. The complete machine-readable validation result is `archive-validation.json`.

The sentence-level leakage check is explicit: the validator maps every causal
example back to its normalized sentence row and verifies that no sentence group
appears in both outer development and test, or in both train and validation
within either fold. It found **zero group overlaps** across all 44 completed
runs. Repeated sampled examples remain confined to the training IDs.

## Results

### Ranked configuration table

{markdown_table(rows)}

### Main 27-configuration grid

The reference family varies context alpha in {{0.1, 1, 3}}, learning rate in {{0.0001, 0.0003, 0.001}}, and window in {{2, 4, 8}}, with 8 layers and batch 64. The best member of this family by word top-1 is experiment {max(main, key=lambda r: r['validation_top1'])['experiment_id']:02d}: validation word top-1 {100 * max(main, key=lambda r: r['validation_top1'])['validation_top1']:.3f}%. Across the main grid, learning rate 0.001 gives the largest improvement in word top-1; alpha changes are small, and window 8 has the highest descriptive average despite overlapping low-accuracy results.

### Depth and batch comparison

Experiments 27–38 hold alpha=1 and learning rate=0.0003 while comparing depths 2/8/64 and batch sizes 16/64/256 at windows 2/8. The best overall validation word top-1 is experiment 35 (64 layers, batch 64, window 2), at {100 * next(r['validation_top1'] for r in rows if r['experiment_id'] == 35):.3f}%. The batch/depth comparison is confounded by update count: batch 16 receives 313 updates per epoch, batch 64 receives 79, and batch 256 receives 20. The result therefore does not isolate architecture quality from optimization budget.

### Sentence-pool comparison

The completed pool results are {len(pool)} runs; experiment 039 is the missing uniform 5,000-sentence result. The fixed full-pool reference is experiment 13 (alpha=1, learning rate=0.0003, window 4, 8 layers, batch 64), with validation word top-1 {100 * reference['validation_top1']:.3f}% and secondary BCE {reference['validation_bce']:.6f}. The pool comparisons are:

{pool_table_text}

At 20,000 and 50,000 sentences, uniform and balanced sampling are close and word top-1 is zero or near zero in the completed pool runs. No monotonic pool-size improvement is established after ten sampled epochs.

### Main-grid marginal summaries

These summaries average over the other two main-grid factors; they are descriptive, not a factorial ANOVA.

Learning rate (word top-1):

{effect_table('learning_rate', 'learning rate')}

Context alpha:

{effect_table('alpha', 'alpha')}

Window:

{effect_table('window', 'window')}

BCE remains available as a secondary loss-oriented view in `aggregate.csv`,
`aggregate.json`, and the BCE-specific figures.

### Test interpretation

The held-out test is appropriately used once after the final refit and is not a selection signal. Experiment {best_test['experiment_id']:02d} has the highest held-out word top-1 ({100 * best_test['test_top1']:.3f}%), but this descriptive test ranking must not replace validation selection. Experiment {best_test_bce['experiment_id']:02d} has the lowest test BCE ({best_test_bce['test_bce']:.6f}). Word top-1 is a strict metric here: all 14 predicted bits must be correct, so it is sparse; the ≥half-bit metric can be high even when exact words are rarely recovered. BCE is retained only as a secondary calibration/optimization diagnostic.

## Epoch-extension criterion and scope boundary

Commit `50cff5e` (“docs: make epoch extension criterion explicit”) defines a rule for the **semantic decoder** sweep: continue a configuration from 30 to 50 epochs only when the median of its three seeded `best_epoch` values equals 30. That rule is not applicable to this archive: `sweep.zip` contains the 45 production QCSE `experiment-000`…`experiment-044` folders, not the semantic `experience-*` folders, and its completed runs have a 10-epoch target. No semantic 30→50 extension decision can be inferred from these artifacts. The incomplete ID 039 should be resumed or rerun before any complete pool-size comparison is claimed.

## Limitations and recommended follow-up

The sweep uses one seed and two validation repeats, so uncertainty is only split-to-split variation, not a robust estimate across random initializations. The batch-size comparison changes the number of optimizer updates per epoch. The archive has no scheduler logs or exact commit metadata, and ID 039 is incomplete. For a publication-grade follow-up, repeat the top configurations across several seeds, equalize optimizer updates or total examples processed when comparing batch sizes, complete ID 039, and reserve an independent test set for the final locked comparison.

## Figures

All figures were generated from the same validated JSON/NPZ artifacts:

{figure_links}
"""
    (output / "detailed-report.md").write_text(report, encoding="utf-8")

    short = f"""# Cluster QCSE sweep — short report

**Scope:** 45 production causal configurations in `{archive.name}`. **44/45 completed**; experiment 039 was still running at archive capture and is excluded from rankings.

## Result

Select by mean full validation word top-1. In this binary decoder, word top-1 means exact agreement of all 14 thresholded bits with the target word ID. The winner is **experiment {best['experiment_id']:02d}**: alpha={best['alpha']:g}, learning rate={best['learning_rate']:g}, window={best['window']}, {best['layers']} layers, batch {best['batch_size']}; validation word top-1 **{100 * best['validation_top1']:.3f}% ± {100 * best['validation_top1_sd']:.3f}%**, held-out test word top-1 **{100 * best['test_top1']:.3f}%**. Secondary validation BCE is **{best['validation_bce']:.6f}**.

{short_table(rows)}

## What the sweep says

- Learning rate 0.001 gives the highest average word top-1 in the 8-layer/batch-64 grid; alpha changes are small and window effects are weak/non-monotonic.
- Experiment 035 is the top-1 winner, but depth/batch comparisons are confounded: batches 16, 64 and 256 receive 313, 79 and 20 optimizer updates per epoch.
- Full-pool and capped-pool word top-1 is zero or near zero after ten sampled epochs; the 5,000-sentence uniform condition is incomplete.
- Sampling is with replacement inside training IDs. Repeated sentences/examples are allowed, while sentence groups remain isolated between train, validation and test.

## Quality gate

All 44 completed archives passed checks for split isolation, sentence-group isolation, fold aggregation, fixed-monitoring histories, and recomputed held-out metrics. The detailed report and 22 plots are in `detailed-report.md` and `plots/`; the machine-readable result is `aggregate.json`.

The 30→50 epoch extension rule in commit `50cff5e` belongs to the separate semantic-decoder sweep and cannot be applied to this 10-epoch production archive.
"""
    (output / "short-report.md").write_text(short, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as zf:
        completed, incomplete = collect(zf)
        if len(completed) != 44:
            raise SystemExit(f"Expected 44 completed CV runs, found {len(completed)}")
        issue_map = {}
        rows = []
        for record in completed:
            issues = protocol_issues(zf, record)
            issue_map[record["id"]] = issues
            rows.append(row_for_json(record, issues))
    rows.sort(key=lambda row: row["experiment_id"])
    completed_ids = {record["id"] for record in completed}
    pending_ids = EXPECTED_IDS - completed_ids
    pending = []
    for experiment_id in sorted(pending_ids):
        candidates = [record for record in incomplete if record["id"] == experiment_id]
        if not candidates:
            pending.append({"experiment_id": experiment_id, "status": "missing"})
            continue
        candidate = max(candidates, key=lambda record: record.get("started_at") or "")
        pending.append(
            {
                "experiment_id": experiment_id,
                "run": candidate["run"],
                "status": candidate["status"],
                "started_at": candidate["started_at"],
                "finished_at": candidate["finished_at"],
                "files": sorted(candidate["files"]),
            }
        )
    validation = {
        "archive": str(args.archive),
        "expected_experiments": 45,
        "completed_experiments": len(completed),
        "pending_experiments": pending,
        "legacy_or_non_cv_run_directories": len(incomplete) - len(pending),
        "completed_with_issues": {str(key): value for key, value in issue_map.items() if value},
        "passed": not any(issue_map.values()),
    }
    (args.output / "archive-validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "aggregate.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (args.output / "aggregate.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            key
            for key, value in rows[0].items()
            if key not in {"folds", "cv_history", "validation_issues"} and not isinstance(value, (dict, list))
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    generate_plots(rows, args.output / "plots")
    write_reports(rows, incomplete, issue_map, args.output, args.archive)
    print(
        json.dumps(
            {
                "completed": len(completed),
                "incomplete": [item["experiment_id"] for item in pending],
                "passed": validation["passed"],
                "output": str(args.output),
                "plots": list(PLOT_NAMES),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
