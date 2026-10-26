"""Audit saved evidence and derive a bound for every dephased readout model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.hamiltonian_study.born_evidence import digest
from qcse.outputs import write_json


def dephased_nll_bound(amplitudes: np.ndarray, rank: int) -> float:
    """A rank-r projector captures at most the r largest eigenvalues of a state."""
    if rank <= 0 or rank > amplitudes.shape[1]:
        raise ValueError("Projector rank must be in the input dimension")
    powers = amplitudes**2
    if not np.allclose(powers.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError("Amplitude states must be normalized")
    maximum_target_probability = np.sort(powers, axis=1)[:, -rank:].sum(1)
    return float(-np.log(maximum_target_probability).mean())


def analyze(run: Path, figure: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lock = json.loads((run / "selection_lock.json").read_text())
    results = json.loads((run / "test_results.json").read_text())
    assert results["selection_lock_sha256"] == digest(run / "selection_lock.json")
    assert lock["source_sha256"] == digest(run / "executed_source.py")
    assert lock["uv_lock_sha256"] == digest(run / "uv.lock")
    assert lock["recipe_lock_sha256"] == digest(run / "recipe_lock.json")
    for name, expected in lock["data"]["files"].items():
        assert digest(run / name) == expected
    for record in lock["fits"]:
        assert record["selected_validation"]["nll"] == min(
            row["validation"]["nll"] for row in record["history"]
        )
        assert record["weights_sha256"] == digest(Path(record["job"]["output"]) / "weights.pt")
    for task in ("modular", "digits"):
        for seed in lock["confirmation_seeds"]:
            paired = [
                row["initial_parameter_hashes"]
                for row in lock["fits"]
                if row["job"]["task"] == task
                and row["job"]["seed"] == seed
                and row["job"]["kind"] in {"coherent", "dephased"}
            ]
            assert len(paired) == 2 and paired[0] == paired[1]
    bounds = {}
    for part in ("train", "validation", "test"):
        with np.load(run / ("digits-test.npz" if part == "test" else "digits-fit.npz")) as data:
            bounds[part] = dephased_nll_bound(data[f"{part}_inputs"], 4)
    checked = 0
    with np.load(run / "test_predictions.npz") as predictions:
        for row in results["fits"]:
            task, kind, seed = row["task"], row["kind"], row["seed"]
            with np.load(run / f"{task}-test.npz") as data:
                targets = data["test_targets"]
            probabilities = predictions[f"{task}-{kind}-s{seed}"]
            nll = float(
                -np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-15)).mean()
            )
            assert abs(nll - row["test"]["nll"]) < 1e-12
            assert np.max(np.abs(probabilities.sum(1) - 1)) < 1e-10
            if task == "digits" and kind == "coherent":
                assert nll < bounds["test"]
            checked += 1

    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    labels = {
        "coherent": "Trainable Hamiltonian",
        "dephased": "Retrained dephased model",
        "mlp": "Classical MLP (similar size)",
    }
    colors = {"coherent": "#146a8a", "dephased": "#bc6632", "mlp": "#427a45"}
    for task, axis in zip(("modular", "digits"), axes, strict=True):
        for kind in labels:
            histories = np.array(
                [
                    [row["validation"]["nll"] for row in record["history"]]
                    for record in lock["fits"]
                    if record["job"]["task"] == task and record["job"]["kind"] == kind
                ]
            )
            epochs = np.arange(histories.shape[1])
            axis.plot(
                epochs, np.maximum(histories.mean(0), 1e-12), label=labels[kind], color=colors[kind]
            )
            axis.fill_between(
                epochs,
                np.maximum(histories.min(0), 1e-12),
                np.maximum(histories.max(0), 1e-12),
                color=colors[kind],
                alpha=0.12,
            )
        axis.set(xlabel="Training epoch", ylabel="Validation NLL (lower is better)")
        axis.grid(alpha=0.18)
    axes[0].set(title="16-symbol composition · 4 qubits", yscale="log")
    axes[1].set_title("UCI digits · 6 qubits")
    axes[1].axhline(
        bounds["validation"],
        color="#555555",
        linestyle="--",
        linewidth=1,
        label="Bound for any dephased model",
    )
    axes[1].legend(fontsize=8, loc="upper right")
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    report = {
        "checked_fits": checked,
        "digit_dephased_nll_lower_bounds": bounds,
        "analysis": "Post-hoc analytic bound; no model selection or further test inference",
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_lock_sha256": results["selection_lock_sha256"],
    }
    write_json(run / "analysis.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, args.figure), indent=2))


if __name__ == "__main__":
    main()
