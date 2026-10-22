"""Mathematical diagnostics and independent circuit checks; no test-set access."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from experiments.encoding_study.simulator import Simulator
from qcse.context import context_matrix, encoding_angles

from .data import graph_matrix, write_json

OUT = Path(__file__).parent / "results"
COLORS = ["#087e8b", "#d45d3c", "#4361a1", "#cf9a35"]


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "#fafafa",
            "axes.facecolor": "#fafafa",
            "savefig.facecolor": "#fafafa",
        }
    )


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def run():
    style()
    rng = np.random.default_rng(19)
    size = 8
    w = rng.random((size, size))
    h = (w + w.T) / 2
    np.fill_diagonal(h, 0)
    h /= np.linalg.norm(h, 2)
    states = normalize(rng.normal(size=(12, size))).astype(complex)
    p = rng.permutation(size)
    time = 2.3
    u = expm(-1j * time * h)
    evolved = states @ u.T
    permuted = states[:, p] @ expm(-1j * time * h[np.ix_(p, p)]).T
    fidelity = np.abs(states @ states.conj().T) ** 2
    evolved_fidelity = np.abs(evolved @ evolved.conj().T) ** 2
    diagnostic: dict[str, Any] = {
        "permutation_max_state_error": float(np.max(np.abs(evolved[:, p] - permuted))),
        "shared_unitary_max_fidelity_change": float(np.max(np.abs(fidelity - evolved_fidelity))),
        "norm_max_error": float(np.max(np.abs(np.linalg.norm(evolved, axis=1) - 1))),
    }
    sparse = expm_multiply(-1j * time * csr_matrix(h), states.T, traceA=0).T
    diagnostic["sparse_vs_dense_max_error"] = float(np.max(np.abs(sparse - evolved)))
    # The normalized Laplacian I-H differs by global phase and the sign of time.
    lap = states @ expm(-1j * time * (np.eye(size) - h)).T
    diagnostic["laplacian_adjacency_born_difference_real_input"] = float(
        np.max(np.abs(abs(lap) ** 2 - abs(evolved) ** 2))
    )
    # Unary XY factor: sum_{i<j} H_ij (XiXj + YiYj)/2, not a double edge sum.
    terms = []
    for i in range(size):
        for j in range(i + 1, size):
            terms += [("XX", [i, j], 0.5 * h[i, j]), ("YY", [i, j], 0.5 * h[i, j])]
    full = SparsePauliOp.from_sparse_list(terms, num_qubits=size).to_matrix()
    indices = 1 << np.arange(size)
    diagnostic["unary_xy_restriction_error"] = float(
        np.max(np.abs(full[np.ix_(indices, indices)] - h))
    )
    unary_u = expm(-1j * time * full)
    diagnostic["unary_evolution_error"] = float(
        np.max(np.abs(unary_u[np.ix_(indices, indices)] - u))
    )
    diagnostic["binary_nonzero_pauli_terms_8_vertices"] = len(
        SparsePauliOp.from_operator(Operator(h)).simplify()
    )
    # Position tensor |r>|w_r>, evolving only words and tracing positions.
    sequence = [1, 3, 5]
    original = np.mean(abs(u[:, sequence]) ** 2, axis=1)
    reversed_p = np.mean(abs(u[:, sequence[::-1]]) ** 2, axis=1)
    diagnostic["uncoupled_position_register_reversal_tv"] = float(
        0.5 * np.abs(original - reversed_p).sum()
    )
    # Directed dilation: conditional target probabilities approach supplied bigram P.
    transition = rng.random((size, size))
    transition /= transition.sum(1, keepdims=True)
    b = np.sqrt(transition)
    spectral_norm = np.linalg.norm(b, 2)
    dilation = np.block([[np.zeros_like(b), b], [b.T, np.zeros_like(b)]]) / spectral_norm
    source = 2
    times = np.geomspace(0.001, 10, 100)
    errors, successes = [], []
    for t in times:
        amplitude = expm(-1j * t * dilation)[size:, source]
        prob = abs(amplitude) ** 2
        successes.append(prob.sum())
        errors.append(0.5 * np.abs(prob / prob.sum() - transition[source]).sum())
    diagnostic["dilation_short_time_tv"] = errors[0]
    diagnostic["dilation_short_time_success"] = successes[0]
    diagnostic["dilation_smalltime_success_ratio"] = successes[0] / (times[0] / spectral_norm) ** 2
    # Encoding geometry under harmless word-ID relabelings; separate from word reversal.
    contexts = rng.integers(0, size, (30, 4))
    sim = Simulator(3)
    encode = lambda rows: sim.encode(  # noqa: E731
        np.array([encoding_angles(context_matrix(c, size), 3) for c in rows]), kind="paper"
    )
    encoded = encode(contexts)
    base_gram = abs(encoded @ encoded.conj().T) ** 2
    drifts = []
    for _ in range(25):
        perm = rng.permutation(size)
        new = encode(perm[contexts])
        drifts.append(float(np.abs(base_gram - abs(new @ new.conj().T) ** 2).mean()))
    diagnostic["qcse_gram_relabeling_mean_absolute_changes"] = drifts
    diagnostic["graph_gram_relabeling_mean_absolute_change"] = float(
        np.abs(evolved_fidelity - abs(permuted @ permuted.conj().T) ** 2).mean()
    )
    write_json(OUT / "theory_diagnostics.json", diagnostic)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].bar(
        ["QCSE\nID angles", "Graph\nHamiltonian"],
        [np.mean(drifts), diagnostic["graph_gram_relabeling_mean_absolute_change"]],
        color=COLORS[:2],
    )
    axes[0].set_ylabel("Mean change in pairwise fidelity")
    axes[0].set_title("Renaming words should do nothing")
    axes[0].text(1, 0.015, "< 10⁻¹⁴", ha="center")
    axes[1].scatter(fidelity.ravel(), evolved_fidelity.ravel(), color=COLORS[0], s=12)
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[1].set(
        xlabel="Fidelity before a shared walk",
        ylabel="Fidelity after the same walk",
        title="A fixed walk preserves geometry",
    )
    axes[2].bar(
        ["Same word\ntwice", "Other word\nonce"],
        [0.8, 0.2],
        width=0.4,
        color=COLORS[0],
        label="Draft: add amplitudes",
    )
    axes[2].scatter(
        [0, 1],
        [2 / 3, 1 / 3],
        marker="D",
        s=55,
        color=COLORS[1],
        label="Alternative: add frequencies",
        zorder=3,
    )
    axes[2].set(
        ylabel="Input measurement probability", ylim=(0, 1), title="Repeated words need a choice"
    )
    axes[2].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "encoding_principles.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].loglog(times, errors, color=COLORS[0])
    axes[0].set(
        xlabel="Evolution time",
        ylabel="Distance to classical bigram (TV)",
        title="Short walks recover the prior",
    )
    axes[1].loglog(times, successes, color=COLORS[1])
    axes[1].set(
        xlabel="Evolution time",
        ylabel="Probability of target-register success",
        title="Low success raises measurement cost",
    )
    fig.tight_layout()
    fig.savefig(OUT / "directed_walk_tradeoff.png", dpi=180)
    plt.close(fig)
    # Physical resource accounting for the real training graph.
    path = OUT / "tatoeba_data" / "dataset.npz"
    if path.exists():
        data = dict(np.load(path))
        resources = {}
        for kind in ["count", "ppmi", "npmi"]:
            graph, weights = graph_matrix(data["cooc"], kind, 24)
            degree = np.count_nonzero(weights, axis=1)
            resources[kind] = {
                "vertices": len(graph),
                "edges": int(np.count_nonzero(graph) // 2),
                "max_degree": int(degree.max()),
                "mean_degree": float(degree.mean()),
                "isolated_vertices": int(np.sum(degree == 0)),
                "binary_qubits": (len(graph) - 1).bit_length(),
                "unary_qubits": len(graph),
                "one_xy_pass_two_qubit_pauli_rotations": int(np.count_nonzero(graph)),
            }
        write_json(OUT / "graph_resources.json", resources)
    return diagnostic


if __name__ == "__main__":
    print({k: v for k, v in run().items() if not isinstance(v, list)})
