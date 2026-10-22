"""Matched three-way comparison using the repository's causal QCSE circuit.

Run prepare, then train, then figure. Selection uses validation only; the final
figure uses the same 8,192 held-out examples for all three configurations.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

from .data import write_json
from .models import HamiltonianQCSEModel
from .run import evaluate, metrics, prediction

ROOT = Path(__file__).parent / "results"
DATA = ROOT / "tatoeba_data"
OUT = ROOT / "causal_comparison"
SEEDS = [11, 22, 33]
MEASURES = ["nll", "top1", "top5"]


def read(path):
    return json.loads(path.read_text())


def prepare():
    """Choose fixed encoding time with the common decoder, then lock the comparison."""
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    if (OUT / "selection_lock.json").exists():
        return read(OUT / "selection_lock.json")
    data = dict(np.load(DATA / "dataset.npz"))
    candidates = []
    for time in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        model = HamiltonianQCSEModel(data, layers=0, evolution_time=time)
        inputs = model.encode(data["validation_contexts"])
        probabilities, _ = prediction(model, inputs)
        score, _ = metrics(probabilities, data["validation_targets"])
        candidates.append({"time": time, "validation": score})
    selected = min(candidates, key=lambda row: row["validation"]["nll"])
    write_json(OUT / "time_selection.json", {"selected": selected, "candidates": candidates})
    selection = {
        "created_utc": datetime.now(UTC).isoformat(),
        "objective": "causal",
        "jobs": [f"causal_hamiltonian_count_L16_s{seed}" for seed in SEEDS],
        "baseline_jobs": [f"qcse_nll_count_L16_s{seed}" for seed in SEEDS],
        "evolution_time": selected["time"],
        "readout": "Z marginals; normalized independent-bit categorical word probabilities",
        "graph": "training-only count graph, 24-neighbor threshold, symmetric normalization",
        "input": "square roots of normalized recency counts, decay 0.7, eight past words",
        "rule": (
            "Fixed evolution time minimizes no-ansatz validation NLL on the declared grid. "
            "Identical fixed Hamiltonian encoder in both Hamiltonian arms. "
            "Same 16-layer forward RX/RZ/CRZ circuit, categorical training loss, "
            "initialization seeds, optimizer and regularization as saved causal QCSE baseline. "
            "Checkpoints selected by validation NLL; no test-based choices. "
            "No unigram mixture or learned readout in any arm."
        ),
        "dataset_sha256": hashlib.sha256((DATA / "dataset.npz").read_bytes()).hexdigest(),
        "scope": "Follow-up requested after the primary study; not an independent fresh test set.",
    }
    write_json(OUT / "selection_lock.json", selection)
    return selection


def train():
    selection = read(OUT / "selection_lock.json")
    command = [
        sys.executable,
        "-m",
        "experiments.hamiltonian_study.run",
        "train",
        "--dataset",
        str(DATA),
        "--output",
        str(OUT),
        "--methods",
        "causal_hamiltonian",
        "--layers",
        "16",
        "--seeds",
        "11",
        "22",
        "33",
        "--epochs",
        "30",
        "--batch-size",
        "128",
        "--workers",
        "3",
        "--evolution-time",
        str(selection["evolution_time"]),
    ]
    subprocess.run(command, check=True)


def figure():
    torch.set_num_threads(1)
    selection = read(OUT / "selection_lock.json")
    data = dict(np.load(DATA / "dataset.npz"))
    assert (
        hashlib.sha256((DATA / "dataset.npz").read_bytes()).hexdigest()
        == (selection["dataset_sha256"])
    )
    if not (OUT / "test_results.json").exists():
        evaluate(OUT, DATA, OUT / "selection_lock.json")
    trained = read(OUT / "test_results.json")["metrics"]
    baseline = read(ROOT / "main" / "test_results.json")["metrics"]
    no_ansatz = HamiltonianQCSEModel(data, layers=0, evolution_time=selection["evolution_time"])
    p, _ = prediction(no_ansatz, no_ansatz.encode(data["test_contexts"]))
    untrained, arrays = metrics(p, data["test_targets"])
    arrays["groups"] = data["test_groups"]
    np.savez_compressed(OUT / "without_ansatz_test.npz", **arrays)
    # Verify the three arms refer to precisely the same targets and sentence groups.
    for directory, keys in [(ROOT / "main", selection["baseline_jobs"]), (OUT, selection["jobs"])]:
        for key in keys:
            saved = np.load(directory / f"{key}_test.npz")
            for field in ["targets", "groups"]:
                assert np.array_equal(saved[field], arrays[field]), (key, field)
            source = baseline if directory == ROOT / "main" else trained
            for metric in MEASURES:
                assert np.isclose(saved[metric].mean(), source[key][metric], atol=1e-12)
    rows = [
        (
            "Baseline QCSE encoding\n+ causal ansatz",
            [baseline[k] for k in selection["baseline_jobs"]],
        ),
        ("Hamiltonian encoding\nwithout ansatz", [untrained]),
        ("Hamiltonian encoding\n+ causal ansatz", [trained[k] for k in selection["jobs"]]),
    ]
    summary = []
    for label, values in rows:
        summary.append(
            {
                "label": label.replace("\n", " "),
                "runs": len(values),
                "mean": {m: float(np.mean([v[m] for v in values])) for m in MEASURES},
                "seed_sd": {
                    m: float(np.std([v[m] for v in values], ddof=1)) if len(values) > 1 else 0.0
                    for m in MEASURES
                },
            }
        )
    write_json(OUT / "three_way_metrics.json", {"selection": selection, "models": summary})
    render_figure(summary, selection)
    return summary


def render_figure(summary, selection, language="en", transparent=False):
    """Render saved measurements without rerunning or changing the experiment."""
    french = language == "fr"
    labels = (
        [
            "Encodage QCSE de référence\n+ ansatz causal",
            "Encodage hamiltonien\nsans ansatz",
            "Encodage hamiltonien\n+ ansatz causal",
        ]
        if french
        else [
            "Baseline QCSE encoding\n+ causal ansatz",
            "Hamiltonian encoding\nwithout ansatz",
            "Hamiltonian encoding\n+ causal ansatz",
        ]
    )
    titles = (
        ["NLL · plus faible = mieux", "Top-1 · plus élevé = mieux", "Top-5 · plus élevé = mieux"]
        if french
        else ["NLL · lower is better", "Top-1 · higher is better", "Top-5 · higher is better"]
    )
    colors = ["#4565A8", "#E6973F", "#098B87"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.9), sharey=True)
    for ax, metric, title in zip(
        axes,
        MEASURES,
        titles,
        strict=True,
    ):
        multiplier = 1 if metric == "nll" else 100
        values = np.array([r["mean"][metric] for r in summary]) * multiplier
        errors = np.array([r["seed_sd"][metric] for r in summary]) * multiplier
        ax.barh(range(3), values, height=0.55, color=colors, zorder=3)
        ax.errorbar(values, range(3), xerr=errors, fmt="none", color="#243442", capsize=4, zorder=4)
        right = float(np.max(values + errors)) * 1.32
        ax.set_xlim(0, right)
        for y, value in enumerate(values):
            label = f"{value:.3f}" if metric == "nll" else f"{value:.2f}%"
            if french:
                label = label.replace(".", ",").replace("%", " %")
            ax.text(
                value + errors[y] + right * 0.025,
                y,
                label,
                va="center",
                weight="bold",
                fontsize=12,
            )
        ax.set_title(title, fontsize=12, weight="bold", pad=16)
        if french:
            ax.set_xlabel(
                "Log-vraisemblance négative" if metric == "nll" else "Mot suivant correct (%)"
            )
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:g}".replace(".", ","))
            )
        else:
            ax.set_xlabel("Negative log likelihood" if metric == "nll" else "Correct next word (%)")
        ax.grid(axis="x", color="#e7ebef", zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=12)
    axes[0].set_yticks(range(3), labels)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Encodage et ansatz causal : comparaison contrôlée"
        if french
        else "Encoding and the causal ansatz: a matched comparison",
        x=0.52,
        y=0.97,
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.52,
        0.90,
        "Même tâche de prédiction causale du mot suivant et même décodage dans les trois cas"
        if french
        else "The same causal next-word task and word-probability decoder in all three",
        ha="center",
        fontsize=11,
        color="#485766",
    )
    time = selection["evolution_time"]
    caption = (
        "Tatoeba · 8 192 cibles de test identiques · "
        "Vocabulaire de 512 mots, dont UNK (mot inconnu)\n"
        "Ansatz : 16 couches, 3 initialisations ; barres d’erreur : écart-type. "
        f"Hamiltonien : exp(−itH) fixé, t = {time:g}. "
        "Décodage : marginales Z → probabilités des mots."
        if french
        else "Tatoeba · 8,192 identical test targets · 512-word vocabulary, including UNK\n"
        "Ansatz: 16 layers, 3 seeds (error bars: seed SD). "
        f"Hamiltonian: fixed exp(−itH), t = {time:g}. "
        "Decoder: Z marginals → word probabilities."
    )
    fig.text(
        0.03,
        0.055,
        caption,
        fontsize=9.3,
        color="#485766",
        linespacing=1.6,
    )
    fig.subplots_adjust(
        left=0.225 if french else 0.205, right=0.985, top=0.79, bottom=0.24, wspace=0.20
    )
    stem = "causal_three_way_fr" if french else "causal_three_way"
    if transparent:
        stem += "_transparent"
    facecolor = "none" if transparent else "white"
    fig.savefig(OUT / f"{stem}.png", dpi=200, facecolor=facecolor, transparent=transparent)
    svg = OUT / f"{stem}.svg"
    fig.savefig(svg, facecolor=facecolor, transparent=transparent)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "figure", "render"])
    parser.add_argument("--language", choices=["en", "fr"], default="en")
    parser.add_argument("--transparent", action="store_true")
    args = parser.parse_args()
    if args.stage == "render":
        saved = read(OUT / "three_way_metrics.json")
        result = render_figure(saved["models"], saved["selection"], args.language, args.transparent)
    else:
        result = {"prepare": prepare, "train": train, "figure": figure}[args.stage]()
    if result is not None:
        print(json.dumps(result))
