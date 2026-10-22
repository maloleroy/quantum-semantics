"""Compact, reproducible summary statistics, figures and validation-only mechanisms."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import write_json
from .models import HamiltonianModel, context_distribution
from .run import make_model, metrics
from .theory import COLORS, style

ROOT = Path(__file__).parent / "results"
LABELS = {
    "unigram": "Unigram",
    "bigram": "Classical bigram",
    "pooled_bigram": "Recency-pooled bigram",
    "walk": "Graph quantum walk",
    "bag": "Literal bag proposal",
    "predictive_residual": "Predictive state + quantum walk",
    "no_graph": "Graph removed (refitted)",
    "shuffled": "Graph labels scrambled (refitted)",
    "diffusion": "Classical undirected diffusion",
    "directed_diffusion": "Classical directed diffusion",
    "embedding": "Classical embedding (8k training)",
    "full_embedding": "Classical embedding (full training)",
    "mixed_bigram": "Bigram + learned unigram mixture",
    "calibrated_bigram": "Classical calibrated bigram",
    "qcse_bce": "QCSE encoding, bit loss",
    "qcse_nll": "QCSE encoding, categorical loss",
    "qcse_born": "QCSE encoding, joint Born readout",
}


def records(stage):
    output = []
    for p in sorted((ROOT / stage).glob("*.json")):
        value = json.loads(p.read_text())
        if "config" in value:
            output.append(value | {"key": p.stem})
    return output


def aggregate_test(stage):
    path = ROOT / stage / "test_results.json"
    if not path.exists():
        return {}
    results = json.loads(path.read_text())["metrics"]
    groups = {}
    for key, value in results.items():
        method = key.split("_count_")[0]
        groups.setdefault(method, []).append(value)
    return {
        method: {
            "runs": len(rows),
            "mean": {k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
            "sd": {
                k: float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0
                for k in rows[0]
            },
        }
        for method, rows in groups.items()
    }


def paired_bootstrap(left_paths, right_paths, metric="nll", repetitions=3000):
    left = [dict(np.load(p)) for p in left_paths]
    right = [dict(np.load(p)) for p in right_paths]
    all_rows = left + right
    assert all(np.array_equal(left[0]["targets"], a["targets"]) for a in all_rows)
    assert all(np.array_equal(left[0]["groups"], a["groups"]) for a in all_rows)
    differences = np.mean([a[metric] for a in left], axis=0) - np.mean(
        [a[metric] for a in right], axis=0
    )
    _, inverse = np.unique(left[0]["groups"], return_inverse=True)
    n_groups = inverse.max() + 1
    sums = np.bincount(inverse, weights=differences)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(761)
    estimates = np.empty(repetitions)
    for r in range(repetitions):
        sample = rng.integers(n_groups, size=n_groups)
        estimates[r] = sums[sample].sum() / sizes[sample].sum()
    return {
        "difference": float(differences.mean()),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "groups": int(n_groups),
        "bootstrap_resamples": repetitions,
    }


def reverse_contexts(contexts):
    reversed_c = contexts.clone()
    for i, row in enumerate(contexts):
        valid = row >= 0
        reversed_c[i, valid] = row[valid].flip(0)
    return reversed_c


def mechanisms(stage="main", dataset="tatoeba_data"):
    """Uses a fixed 256 validation examples, not test-set behavior."""
    torch.set_num_threads(1)
    data = dict(np.load(ROOT / dataset / "dataset.npz"))
    diagnostic_ids = np.sort(
        np.random.default_rng(702).choice(
            len(data["validation_targets"]),
            min(256, len(data["validation_targets"])),
            replace=False,
        )
    )
    contexts = torch.tensor(data["validation_contexts"][diagnostic_ids])
    targets = data["validation_targets"][diagnostic_ids]
    result = {}
    for record in records(stage):
        method = record["config"]["method"]
        if method not in ("walk", "predictive_residual", "shuffled", "no_graph"):
            continue
        model = make_model(data, record["config"])
        assert isinstance(model, HamiltonianModel)
        model.load_state_dict(
            torch.load(ROOT / stage / f"{record['key']}.pt", weights_only=True), strict=False
        )
        with torch.no_grad():
            p, aux = model(contexts, return_aux=True)
            reversed_p = model(reverse_contexts(contexts))
            eigenvalues = model.eigenvalues.clone()
            model.eigenvalues.zero_()
            removed = model(contexts)
            model.eigenvalues.copy_(eigenvalues)
            # Clear phases after each fitted walk: identical beta, stochastic |U|^2 kernels.
            c = context_distribution(contexts, model.size)
            dephased = model.transition[contexts[:, -1]] if method == "predictive_residual" else c
            identity = torch.eye(model.size, dtype=torch.complex64)
            for beta in model.beta:
                kernel = model.evolve(identity, beta).abs().square()
                dephased = dephased @ kernel
                dephased = dephased / dephased.sum(1, keepdim=True)
            eta = model.mixture.sigmoid()
            dephased = (1 - eta) * dephased + eta * model.unigram
            state = aux["state"]
            initial = model.transition[contexts[:, -1]] if method == "predictive_residual" else c
            initial = initial.sqrt().to(torch.complex64)
            before = (initial @ initial.conj().T).abs().square()
            after = (state @ state.conj().T).abs().square()
            raw_target = aux["raw"][np.arange(len(contexts)), targets].numpy()
        base, _ = metrics(p.numpy(), targets)
        no_graph, _ = metrics(removed.numpy(), targets)
        classicalized, _ = metrics(dephased.numpy(), targets)
        result[record["key"]] = {
            "validation_subset_nll": base["nll"],
            "fixed_weights_graph_removal_nll_change": no_graph["nll"] - base["nll"],
            "fixed_weights_dephasing_nll_change": classicalized["nll"] - base["nll"],
            "mean_reversal_tv": float(0.5 * (p - reversed_p).abs().sum(1).mean()),
            "mean_fidelity_geometry_change": float((before - after).abs().mean()),
            "unigram_mixture": float(eta),
            "raw_born_nll_clipped_1e30": float(-np.log(np.maximum(raw_target, 1e-30)).mean()),
            "raw_target_zero_rate": float(np.mean(raw_target <= 1e-30)),
            "state_norm_max_error": float((state.abs().square().sum(1) - 1).abs().max()),
        }
    write_json(ROOT / "mechanisms.json", result)
    return result


def make_tables(summary):
    lines = [
        "| Model | Test NLL | Perplexity | Top-1 | Top-5 | Known-target NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, row in sorted(summary.items(), key=lambda kv: kv[1]["mean"]["nll"]):
        v, sd = row["mean"], row["sd"]
        uncertainty = f" ± {sd['nll']:.4f}" if row["runs"] > 1 else ""
        lines.append(
            f"| {LABELS.get(key, key)} | {v['nll']:.4f}{uncertainty} | "
            f"{v['perplexity']:.2f} | {100 * v['top1']:.2f}% | "
            f"{100 * v['top5']:.2f}% | {v['known_nll']:.4f} |"
        )
    (ROOT / "test_table.md").write_text("\n".join(lines) + "\n")


def figures(summary):
    style()
    featured = {
        "qcse_nll",
        "qcse_born",
        "walk",
        "diffusion",
        "predictive_residual",
        "mixed_bigram",
        "bigram",
        "full_embedding",
    }
    ordered = sorted(set(summary) & featured, key=lambda k: summary[k]["mean"]["nll"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
    quantum = {"walk", "predictive_residual", "no_graph", "shuffled"}
    for i, key in enumerate(ordered):
        row = summary[key]
        color = COLORS[0] if key in quantum else COLORS[2] if key.startswith("qcse") else COLORS[1]
        for ax, metric, scale in zip(axes, ["nll", "top1"], [1, 100], strict=True):
            ax.errorbar(
                row["mean"][metric] * scale,
                i,
                xerr=row["sd"][metric] * scale,
                color=color,
                marker="o",
                capsize=3,
                markersize=6,
            )
    axes[0].set_yticks(range(len(ordered)), [LABELS.get(k, k) for k in ordered])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Negative log likelihood (lower is better)", title="Probability quality")
    axes[1].set(xlabel="Correct next word (%)", title="Exact top-1 prediction")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Held-out Tatoeba: corpus statistics help; quantum gain needs its own control",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.text(
        0.5,
        0.02,
        "8,192 held-out targets · 512-word vocabulary including UNK · "
        "error bars: SD across 3 optimization seeds",
        ha="center",
        fontsize=9,
    )
    fig.savefig(ROOT / "heldout_comparison.png", dpi=180)
    plt.close(fig)
    all_records = records("depth") + [r for r in records("main") if r["config"]["method"] == "walk"]
    by_depth = {}
    for r in all_records:
        by_depth.setdefault(r["config"]["layers"], []).append(r["validation"]["nll"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    xs = sorted(by_depth)
    ys = np.array([np.mean(by_depth[x]) for x in xs])
    sd = np.array([np.std(by_depth[x], ddof=1) for x in xs])
    axes[0].errorbar(xs, ys, yerr=sd, fmt="o-", color=COLORS[0], capsize=4)
    axes[0].set(
        xlabel="Alternating graph/context layers",
        ylabel="Validation NLL",
        xticks=xs,
        title="Depth sweep: same data and training epochs",
    )
    for method, color in [
        ("walk", COLORS[0]),
        ("predictive_residual", COLORS[3]),
        ("qcse_nll", COLORS[2]),
        ("embedding", COLORS[1]),
    ]:
        history = [r["history"] for r in records("main") if r["config"]["method"] == method]
        if history:
            epochs = [r["epoch"] for r in history[0]]
            loss = np.mean([[r["validation_nll"] for r in h] for h in history], axis=0)
            axes[1].plot(epochs, loss, color=color, label=LABELS[method])
    axes[1].set(xlabel="Epoch", ylabel="Validation NLL", title="What training changes")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "depth_and_learning.png", dpi=180)
    plt.close(fig)


def main():
    summary = aggregate_test("main")
    summary.update(aggregate_test("calibration"))
    born = aggregate_test("born_control")
    if "qcse_born" in born:
        summary["qcse_born"] = born["qcse_born"]
    full = aggregate_test("full_classical")
    if "embedding" in full:
        summary["full_embedding"] = full["embedding"]
    write_json(ROOT / "summary.json", summary)
    make_tables(summary)
    figures(summary)
    comparisons = {}
    for left, right in [
        ("walk", "qcse_nll"),
        ("walk", "no_graph"),
        ("walk", "shuffled"),
        ("walk", "diffusion"),
        ("predictive_residual", "bigram"),
    ]:
        lp = list((ROOT / "main").glob(f"{left}_count_L16_s*_test.npz"))
        rp = list((ROOT / "main").glob(f"{right}_count_L16_s*_test.npz"))
        if right == "bigram":
            rp = [ROOT / "main" / "bigram_test.npz"]
        comparisons[f"{left} - {right}"] = {
            metric: paired_bootstrap(lp, rp, metric) for metric in ["nll", "top1"]
        }
    if full:
        comparisons["full_embedding - bigram"] = {
            metric: paired_bootstrap(
                list((ROOT / "full_classical").glob("embedding_*_test.npz")),
                [ROOT / "main" / "bigram_test.npz"],
                metric,
            )
            for metric in ["nll", "top1"]
        }
    for control in ["mixed_bigram", "calibrated_bigram"]:
        paths = list((ROOT / "calibration").glob(f"{control}_count_L0_s*_test.npz"))
        if paths:
            comparisons[f"predictive_residual - {control}"] = {
                metric: paired_bootstrap(
                    list((ROOT / "main").glob("predictive_residual_count_L16_s*_test.npz")),
                    paths,
                    metric,
                )
                for metric in ["nll", "top1"]
            }
    if born:
        comparisons["walk - qcse_born"] = {
            metric: paired_bootstrap(
                list((ROOT / "main").glob("walk_count_L16_s*_test.npz")),
                list((ROOT / "born_control").glob("qcse_born_count_L16_s*_test.npz")),
                metric,
            )
            for metric in ["nll", "top1"]
        }
    write_json(ROOT / "paired_comparisons.json", comparisons)
    mechanisms()


if __name__ == "__main__":
    main()
