"""A fixed small pilot: paired seeds, all outcomes saved, no test selection."""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import qiskit

from qcse.data import build_vocabulary, load_phrases, make_examples, split_examples, word_bits
from qcse.model import QCSEModel
from qcse.training import TrainConfig, metrics, train
from research.semantics import code_geometry, fit_linear, fit_projection, semantic_codes
from research.variants import ResearchModel, Variant, batch_evolve

SEEDS = [11, 23, 37]
MAP_SEEDS = [101, 202, 303]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def select_data(vocabulary, window):
    sentences = load_phrases()
    all_examples = make_examples(sentences, vocabulary, window)
    train_ids, test_ids = split_examples(all_examples, sentences, seed=31415)
    rng = np.random.default_rng(2718)
    selected = []
    for ids, count in [(train_ids, 24), (test_ids, 8)]:
        groups = sorted({tuple(sentences[all_examples[i].sentence]) for i in ids})
        chosen = {groups[i] for i in rng.choice(len(groups), count, replace=False)}
        selected.extend(i for i in ids if tuple(sentences[all_examples[i].sentence]) in chosen)
    original = np.array(sorted(selected))
    examples = [all_examples[i] for i in original]
    is_train = np.isin(original, train_ids)
    return examples, np.flatnonzero(is_train), np.flatnonzero(~is_train), original


def scores(probabilities, targets, canonical_ids, codes):
    row = metrics(probabilities, targets)
    predicted = (probabilities >= 0.5) @ (1 << np.arange(targets.shape[1]))
    row["invalid_code_rate"] = float((~np.isin(predicted, codes)).mean())
    # Factorized marginal likelihood decoder; does NOT claim to be joint Born probabilities.
    code_bits = word_bits(codes, targets.shape[1])
    p = np.clip(probabilities, 1e-10, 1 - 1e-10)
    likelihood = np.log(p) @ code_bits.T + np.log1p(-p) @ (1 - code_bits).T
    row["valid_code_word_accuracy"] = float((likelihood.argmax(axis=1) == canonical_ids).mean())
    return row


def classical_controls(examples, train_ids, test_ids, codes, projected=None):
    q = (len(codes) - 1).bit_length()
    ids = np.array([e.target for e in examples])
    y = word_bits(codes[ids], q)
    prior = (y[train_ids].sum(axis=0) + 1) / (len(train_ids) + 2)
    bit_means = np.array([word_bits(codes[list(e.context)], q).mean(axis=0) for e in examples])
    predictions = {"train_bit_prior": np.tile(prior, (len(y), 1)), "context_bit_mean": bit_means}
    predictions["linear_bit_mean"], _ = fit_linear(bit_means, y, train_ids)
    if projected is not None:
        features = np.array([projected[list(e.context)].mean(axis=0) for e in examples])
        predictions["linear_qwen_projected"], _ = fit_linear(features, y, train_ids)
    return {
        name: {
            "train": scores(p[train_ids], y[train_ids], ids[train_ids], codes),
            "test": scores(p[test_ids], y[test_ids], ids[test_ids], codes),
        }
        for name, p in predictions.items()
    }, predictions


def run_condition(
    out,
    variant,
    seed,
    vocabulary,
    examples,
    train_ids,
    test_ids,
    codes,
    vectors,
    projected,
    config,
    shuffle_labels=False,
):
    name = f"{variant.name}-s{seed}"
    folder = out / name
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir()
    canonical_ids = np.array([e.target for e in examples])
    coded = [replace(e, target=int(codes[e.target])) for e in examples]
    if shuffle_labels:
        shuffled = np.random.default_rng(515).permutation([coded[i].target for i in train_ids])
        for i, target in zip(train_ids, shuffled):
            coded[i] = replace(coded[i], target=int(target))
    model = ResearchModel(
        vocabulary, variant, vectors=vectors, projected=projected, window=8, layers=2, seed=seed
    )
    start = time.perf_counter()
    initial = model.weights.copy()
    cfg = replace(config, seed=seed)
    history, probabilities = train(model, coded, train_ids, test_ids, cfg)
    # Evaluate against REAL labels, even in shuffled-label negative control.
    targets = word_bits(codes[canonical_ids], model.qubits)
    layers = Counter(len(model.angles(e.context)) for e in examples)
    oracle_error = 0.0
    for i in test_ids[:3]:
        state = model.encode(examples[i].context).evolve(model.bound_ansatz())
        expected = state.probabilities() @ model._basis_bits
        oracle_error = max(oracle_error, float(np.max(np.abs(expected - probabilities[i]))))
    row = {
        "name": variant.name,
        "seed": seed,
        "variant": asdict(variant),
        "training": asdict(cfg),
        "shuffle_training_labels": shuffle_labels,
        "seconds": time.perf_counter() - start,
        "qubits": model.qubits,
        "parameter_slots": len(model.parameters),
        "circuit_parameters": len(model.ansatz.parameters),
        "encoding_layers_histogram": dict(sorted(layers.items())),
        "test_oracle_max_error": oracle_error,
        "weights_l2_change": float(np.linalg.norm(model.weights - initial)),
        "initial": history[0],
        "final_training_history": history[-1],
        "train": scores(
            probabilities[train_ids], targets[train_ids], canonical_ids[train_ids], codes
        ),
        "test": scores(probabilities[test_ids], targets[test_ids], canonical_ids[test_ids], codes),
    }
    np.savez_compressed(
        folder / "predictions.npz",
        probabilities=probabilities,
        targets=targets,
        canonical_target_ids=canonical_ids,
        codebook=codes,
        weights=model.weights,
        initial_weights=initial,
        train_ids=train_ids,
        test_ids=test_ids,
    )
    write_json(folder / "history.json", history)
    write_json(folder / "result.json", row)
    print(
        f"{name}: BCE {row['test']['bce']:.4f}; exact {row['test']['exact_word_accuracy']:.3f}; "
        f"{row['seconds']:.1f}s",
        flush=True,
    )
    return row


def audit(out):
    out.mkdir(parents=True, exist_ok=False)
    vocabulary = build_vocabulary(load_phrases())
    examples, train_ids, test_ids, original = select_data(vocabulary, 4)
    # A full 50-epoch miniature run at the paper's learning rate and penalty.
    model = QCSEModel(vocabulary, layers=2, seed=11)
    start = time.perf_counter()
    history, predictions = train(
        model, examples, train_ids, test_ids, TrainConfig(epochs=50, seed=11)
    )
    targets = word_bits([e.target for e in examples], model.qubits)
    controls, _ = classical_controls(examples, train_ids, test_ids, np.arange(len(vocabulary)))
    context = examples[0].context
    state = model.encode(context)
    eps = 1e-5
    gradient = []
    for i in range(len(model.weights)):
        plus, minus = model.weights.copy(), model.weights.copy()
        plus[i] += eps
        minus[i] -= eps
        diff = (model.predict_encoded([state], plus) - model.predict_encoded([state], minus)) / (
            2 * eps
        )
        gradient.append(float(np.linalg.norm(diff)))
    # Independent commuting-CRZ order control, keeping edge-to-angle association.
    from qiskit import QuantumCircuit

    c = model.bound_ansatz()
    reversed_crz = QuantumCircuit(model.qubits)
    pending = []
    for item in c.data:
        if item.operation.name == "crz":
            pending.append(item)
        else:
            for previous in reversed(pending):
                reversed_crz.append(previous.operation, previous.qubits)
            pending = []
            reversed_crz.append(item.operation, item.qubits)
    for previous in reversed(pending):
        reversed_crz.append(previous.operation, previous.qubits)
    commutation_error = float(
        np.max(np.abs(state.evolve(c).data - state.evolve(reversed_crz).data))
    )
    batch_error = float(np.max(np.abs(batch_evolve([state], c)[0] - state.evolve(c).data)))
    q = model.qubits
    summary = {
        "vocabulary": len(vocabulary),
        "qubits": q,
        "train_examples": len(train_ids),
        "test_examples": len(test_ids),
        "training": asdict(TrainConfig(epochs=50, seed=11)),
        "seconds": time.perf_counter() - start,
        "initial": history[0],
        "final": history[-1],
        "controls": controls,
        "single_context_probability_jacobian_column_norms": gradient,
        "commuting_crz_max_state_error": commutation_error,
        "batch_max_state_error": batch_error,
        "fair_random_bits_expected_paper_accuracy": sum(
            math.comb(q, k) for k in range(math.ceil(q / 2), q + 1)
        )
        / 2**q,
        "fair_random_bits_expected_exact_accuracy": 1 / 2**q,
        "paper_sha256": hashlib.sha256(Path("../paper4.pdf").read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    write_json(out / "result.json", summary)
    write_json(out / "history.json", history)
    np.savez_compressed(
        out / "predictions.npz",
        probabilities=predictions,
        targets=targets,
        train_ids=train_ids,
        test_ids=test_ids,
        original_example_ids=original,
        weights=model.weights,
        vocabulary=vocabulary,
    )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/pilot"))
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        audit(args.output)
        return
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    with np.load("artifacts/qwen/vocabulary.npz") as archive:
        vocabulary = archive["vocabulary"].tolist()
        vectors = archive["vectors"].astype(float)
    examples, train_ids, test_ids, original = select_data(vocabulary, 8)
    q = (len(vocabulary) - 1).bit_length()
    canonical = np.arange(len(vocabulary))
    semantic = semantic_codes(vectors, q)
    random_semantic = np.random.default_rng(101).permutation(semantic)
    projected, projection = fit_projection(vectors, examples, train_ids, 2 * q)
    shuffled_vectors = vectors[np.random.default_rng(101).permutation(len(vectors))]
    config = TrainConfig(epochs=20, batch_size=32, learning_rate=0.003, l2=0.001)
    conditions = [(Variant(), canonical, vectors, False)]
    for map_seed in MAP_SEEDS:
        conditions.append(
            (
                Variant(f"id-map-{map_seed}", input_ids="fixed", map_seed=map_seed),
                canonical,
                vectors,
                False,
            )
        )
    conditions += [
        (
            Variant("joint-id-map", input_ids="fixed"),
            np.random.default_rng(101).permutation(canonical),
            vectors,
            False,
        ),
        (Variant("fixed-pairs", layout="fixed_pairs"), canonical, vectors, False),
        (Variant("layer-pairs", layout="layer_pairs"), canonical, vectors, False),
        (Variant("reverse-uploads", layout="reverse_layers"), canonical, vectors, False),
        (Variant("layer-id-maps", input_ids="per_layer"), canonical, vectors, False),
        (Variant("rz-before-rx", rotations="rz_rx"), canonical, vectors, False),
        (Variant("reverse-cnot", reverse_cnot=True), canonical, vectors, False),
        (Variant("shuffled-labels"), canonical, vectors, True),
        (Variant("semantic-codes"), semantic, vectors, False),
        (Variant("random-codes"), random_semantic, vectors, False),
        (Variant("qwen-layout", layout="semantic"), canonical, vectors, False),
        (Variant("shuffled-qwen-layout", layout="semantic"), canonical, shuffled_vectors, False),
        (
            Variant("qwen-layout-no-entanglement", layout="semantic", entangle=False),
            canonical,
            vectors,
            False,
        ),
        (
            Variant("shuffled-layout-no-entanglement", layout="semantic", entangle=False),
            canonical,
            shuffled_vectors,
            False,
        ),
        (Variant("qwen-direct", direct=True), canonical, vectors, False),
    ]
    protocol = {
        "seeds": SEEDS,
        "map_seeds": MAP_SEEDS,
        "training": asdict(config),
        "conditions": [asdict(v) for v, _, _, _ in conditions],
        "runs": len(conditions) * len(SEEDS),
        "train_examples": len(train_ids),
        "test_examples": len(test_ids),
        "train_sentence_groups": 24,
        "test_sentence_groups": 8,
        "split_seed": 31415,
        "selection_seed": 2718,
        "window": 8,
        "ansatz_layers": 2,
        "vocabulary_size": len(vocabulary),
        "vocabulary_policy": "alphabetical full corpus, transductive dictionary",
        "corpus_sha256": hashlib.sha256(Path("phrases.csv").read_bytes()).hexdigest(),
        "ideas_sha256": hashlib.sha256(Path("../QCSE_ideas.md").read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "qiskit": qiskit.__version__,
        "primary_metric": "held-out bit BCE, no checkpoint or hyperparameter selection",
        "interpretation": "paired initialization variability on one tiny fixed split; no significance claims",
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "examples.json", [asdict(e) for e in examples])
    write_json(out / "vocabulary.json", vocabulary)
    np.savez_compressed(
        out / "data.npz",
        original_example_ids=original,
        train_ids=train_ids,
        test_ids=test_ids,
        semantic_codes=semantic,
        random_codes=random_semantic,
        projected_vocab=projected,
        **projection,
    )
    geometry = {
        name: code_geometry(vectors, codes, q)
        for name, codes in [
            ("alphabetical", canonical),
            ("semantic", semantic),
            ("shuffled_semantic", random_semantic),
        ]
    }
    write_json(out / "code_geometry.json", geometry)
    controls = {}
    for name, codes in [
        ("canonical", canonical),
        ("semantic", semantic),
        ("random", random_semantic),
        ("joint", np.random.default_rng(101).permutation(canonical)),
    ]:
        controls[name], predictions = classical_controls(
            examples, train_ids, test_ids, codes, projected
        )
        np.savez_compressed(out / f"classical-{name}.npz", **predictions)
    write_json(out / "classical.json", controls)
    results = []
    for variant, codes, layout_vectors, shuffle in conditions:
        for seed in SEEDS:
            results.append(
                run_condition(
                    out,
                    variant,
                    seed,
                    vocabulary,
                    examples,
                    train_ids,
                    test_ids,
                    codes,
                    layout_vectors,
                    projected,
                    config,
                    shuffle,
                )
            )
            write_json(out / "results.json", results)


if __name__ == "__main__":
    main()
