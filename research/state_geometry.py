"""Check what semantic codes do (and cannot do) to full quantum geometry."""

import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.stats import spearmanr

from qcse.data import word_bits
from research.variants import ResearchModel, Variant, batch_evolve


def main():
    root = Path("artifacts/pilot")
    vocabulary = json.loads((root / "vocabulary.json").read_text())
    examples = json.loads((root / "examples.json").read_text())
    with np.load(root / "data.npz") as d:
        test_ids = d["test_ids"]
        codebooks = {
            "alphabetical": np.arange(len(vocabulary)),
            "semantic": d["semantic_codes"],
            "shuffled_semantic": d["random_codes"],
        }
    with np.load("artifacts/qwen/vocabulary.npz") as d:
        vectors = d["vectors"].astype(float)
    targets = np.array([examples[i]["target"] for i in test_ids])
    semantic_similarity = vectors[targets] @ vectors[targets].T
    pairs = np.triu_indices(len(test_ids), 1)
    results = []
    for name in ["canonical", "semantic-codes", "random-codes"]:
        for seed in [11, 23, 37]:
            folder = root / f"{name}-s{seed}"
            with np.load(folder / "predictions.npz") as d:
                probabilities = d["probabilities"][test_ids]
                weights = d["weights"]
            model = ResearchModel(vocabulary, Variant(name), window=8, seed=seed)
            model.weights = weights
            states = [model.encode(examples[i]["context"]) for i in test_ids]
            before = np.array([s.data for s in states])
            after = batch_evolve(states, model.bound_ansatz())
            gram_before = np.abs(before.conj() @ before.T) ** 2
            gram_after = np.abs(after.conj() @ after.T) ** 2
            distance = np.linalg.norm(probabilities[:, None] - probabilities[None, :], axis=2)
            results.append(
                {
                    "condition": name,
                    "seed": seed,
                    "max_fidelity_change_under_training_unitary": float(
                        np.max(np.abs(gram_before - gram_after))
                    ),
                    "target_cosine_vs_full_state_fidelity_spearman": float(
                        spearmanr(semantic_similarity[pairs], gram_after[pairs]).statistic
                    ),
                    "target_cosine_vs_negative_marginal_distance_spearman": float(
                        spearmanr(semantic_similarity[pairs], -distance[pairs]).statistic
                    ),
                }
            )
    # Illustrate a genuine nonorthogonal semantic state construction, separate from QCSE.
    # Binary basis states are orthogonal for ANY different code; Ry(delta*b) product
    # states have fidelity cos(delta/2)^(2*Hamming), so nearby codes now mean close states.
    delta = np.pi / 3
    product = {}
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argsort(-similarity, axis=1)[:, :5]
    for name, codes in codebooks.items():
        bits = word_bits(codes, 10)
        hamming = np.count_nonzero(bits[:, None] != bits[neighbors], axis=2)
        analytic = np.cos(delta / 2) ** (2 * hamming)
        checks = []
        for i in [0, 10, 100, 500]:
            j = neighbors[i, 0]
            states = []
            for k in [i, j]:
                circuit = QuantumCircuit(10)
                for q, bit in enumerate(bits[k]):
                    circuit.ry(float(delta * bit), q)
                states.append(Statevector.from_instruction(circuit))
            checks.append(abs(abs(np.vdot(states[0].data, states[1].data)) ** 2 - analytic[i, 0]))
        product[name] = {
            "nearest5_mean_product_state_fidelity": float(analytic.mean()),
            "qiskit_formula_max_error": float(max(checks)),
            "delta_radians": float(delta),
        }
    result = {
        "note": "Pairwise correlations share words/sentences: descriptive, no independent-pair p-values.",
        "shared_unitary_geometry": results,
        "separate_nonorthogonal_product_code_states": product,
        "interpretation": "Shared ansatz cannot change full-state fidelity. Product-code construction is a separate representation diagnostic, not trained QCSE or a quantum advantage.",
    }
    (root / "state_geometry.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
