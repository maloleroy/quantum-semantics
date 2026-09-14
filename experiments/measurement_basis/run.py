"""Compare Z and learned local readout bases for causal QCSE.

Run from QCSE with ``uv run python -m experiments.measurement_basis.run --help``.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from qcse.context import context_matrix, encoding_angles
from qcse.data import DEFAULT_DATA, build_vocabulary, load_phrases, make_examples, split_examples
from qcse.training import binary_cross_entropy, metrics

from .simulator import Simulator

METHODS = ("z", "learned_y", "learned_xyz")


def vocabulary_metrics(probabilities, targets, bits):
    probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
    log_scores = np.log(probabilities) @ bits.T + np.log1p(-probabilities) @ (1 - bits).T
    row_max = log_scores.max(axis=1, keepdims=True)
    log_normalizer = row_max[:, 0] + np.log(np.exp(log_scores - row_max).sum(axis=1))
    ranks = np.argsort(-log_scores, axis=1, kind="stable")
    result = metrics(probabilities, bits[targets])
    result.update(
        {
            "vocab_top1": float(np.mean(ranks[:, 0] == targets)),
            "vocab_top5": float(np.mean(np.any(ranks[:, :5] == targets[:, None], axis=1))),
            "vocab_nll": float(
                np.mean(log_normalizer - log_scores[np.arange(len(targets)), targets])
            ),
            "invalid_id_rate": float(
                np.mean(
                    (probabilities >= 0.5) @ (1 << np.arange(probabilities.shape[1]))
                    >= len(bits)
                )
            ),
        }
    )
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare(args):
    sentences = load_phrases(args.data)
    vocabulary = build_vocabulary(sentences)
    examples = make_examples(sentences, vocabulary, args.window, "causal")
    outer_train, test = split_examples(examples, sentences, seed=42)
    inside = [examples[i] for i in outer_train]
    local_train, local_val = split_examples(inside, sentences, seed=31415, test_fraction=0.2)
    train, val = outer_train[local_train], outer_train[local_val]
    rng = np.random.default_rng(1618)
    if args.train_limit:
        train = np.sort(rng.choice(train, min(len(train), args.train_limit), replace=False))
    if args.val_limit:
        val = np.sort(rng.choice(val, min(len(val), args.val_limit), replace=False))
    return sentences, vocabulary, examples, train, val, test


def encode_all(sim, examples, vocabulary_size):
    """Encode variable-length causal contexts without adding non-identity padding layers."""
    grouped = {}
    for index, example in enumerate(examples):
        angles = encoding_angles(context_matrix(example.context, vocabulary_size), sim.qubits)
        grouped.setdefault(len(angles), []).append((index, angles))
    states = np.empty((len(examples), sim.dim), dtype=complex)
    for rows in grouped.values():
        for start in range(0, len(rows), 128):
            block = rows[start : start + 128]
            ids = np.array([row[0] for row in block])
            angles = np.stack([row[1] for row in block])
            states[ids] = sim.encode(angles)
    return states


def readout_count(method, qubits):
    return {"z": 0, "learned_y": qubits, "learned_xyz": 2 * qubits}[method]


def readout_summary(method, angles, qubits):
    if method == "z":
        return {"angles": [], "mean_tilt_degrees": 0.0, "max_tilt_degrees": 0.0}
    if method == "learned_y":
        measured_z = np.cos(angles)
    else:
        rx, ry = angles.reshape(qubits, 2).T
        measured_z = np.cos(rx) * np.cos(ry)
    tilts = np.degrees(np.arccos(np.clip(measured_z, -1, 1)))
    return {
        "angles": angles.tolist(),
        "mean_tilt_degrees": float(tilts.mean()),
        "max_tilt_degrees": float(tilts.max()),
    }


def fit_one(job):
    args_dict, method, seed = job
    args = argparse.Namespace(**args_dict)
    output = Path(args.output)
    result_path = output / f"{method}_seed{seed}.json"
    if result_path.exists() and not args.overwrite:
        return {"method": method, "seed": seed, "cached": True}
    start_time = time.perf_counter()
    _, vocabulary, examples, train, val, test = prepare(args)
    qubits = (len(vocabulary) - 1).bit_length()
    sim = Simulator(qubits)
    states = encode_all(sim, examples, len(vocabulary))
    targets = np.array([example.target for example in examples])
    target_bits = ((targets[:, None] >> np.arange(qubits)) & 1).astype(float)
    vocabulary_bits = (
        (np.arange(len(vocabulary))[:, None] >> np.arange(qubits)) & 1
    ).astype(float)
    n_ansatz = args.layers * (3 * qubits - 1)
    n_readout = readout_count(method, qubits)
    rng = np.random.default_rng(seed)
    # The ansatz initialization is identical for all methods; zero readout angles
    # make both learned models exactly equal to the Z baseline at epoch zero.
    weights = np.r_[rng.uniform(-np.pi, np.pi, n_ansatz), np.zeros(n_readout)]

    def evolved(ids, ansatz_weights):
        return sim.ansatz(states[ids], ansatz_weights)

    def predict(ids, current_weights):
        state = evolved(ids, current_weights[:n_ansatz])
        return sim.readout_probabilities(state, method, current_weights[n_ansatz:])

    def predict_all(ids, current_weights):
        return np.concatenate(
            [
                predict(ids[start : start + 128], current_weights)
                for start in range(0, len(ids), 128)
            ]
        )

    initial_val = vocabulary_metrics(predict_all(val, weights), targets[val], vocabulary_bits)
    history = [{"epoch": 0, "validation": initial_val}]
    best_loss, best_epoch, best = initial_val["bce"], 0, weights.copy()
    first, second, step = np.zeros_like(weights), np.zeros_like(weights), 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train)
        for start in range(0, len(order), args.batch_size):
            ids = order[start : start + args.batch_size]
            step += 1
            delta = rng.choice([-1.0, 1.0], size=n_ansatz)
            c = args.perturbation / step**0.101
            readout_angles = weights[n_ansatz:]
            plus_state = evolved(ids, weights[:n_ansatz] + c * delta)
            plus_p = sim.readout_probabilities(plus_state, method, readout_angles)
            plus_loss = binary_cross_entropy(plus_p, target_bits[ids])
            if n_readout:
                plus_readout_gradient = sim.readout_gradient(
                    plus_state, target_bits[ids], method, readout_angles
                )
            minus_state = evolved(ids, weights[:n_ansatz] - c * delta)
            minus_p = sim.readout_probabilities(minus_state, method, readout_angles)
            minus_loss = binary_cross_entropy(minus_p, target_bits[ids])
            ansatz_gradient = (plus_loss - minus_loss) / (2 * c) * delta
            ansatz_gradient += 2 * args.l2 * weights[:n_ansatz]
            if n_readout:
                minus_readout_gradient = sim.readout_gradient(
                    minus_state, target_bits[ids], method, readout_angles
                )
                basis_gradient = 0.5 * (plus_readout_gradient + minus_readout_gradient)
                basis_gradient += 2 * args.l2 * readout_angles
                gradient = np.r_[ansatz_gradient, basis_gradient]
            else:
                gradient = ansatz_gradient
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            weights -= (
                args.learning_rate
                * first
                / (1 - 0.9**step)
                / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            values = vocabulary_metrics(predict_all(val, weights), targets[val], vocabulary_bits)
            history.append({"epoch": epoch, "validation": values})
            if values["bce"] < best_loss:
                best_loss, best_epoch, best = values["bce"], epoch, weights.copy()

    partitions = {"train": train, "validation": val}
    if args.test:
        partitions["test"] = test
    results = {}
    diagnostics = {}
    payload = {"weights": best, "vocabulary": np.array(vocabulary)}
    width = 3 * qubits - 1
    no_last_diagonal = best[:n_ansatz].copy()
    final = (args.layers - 1) * width
    no_last_diagonal[final + 1 : final + 2 * qubits : 2] = 0
    no_last_diagonal[final + 2 * qubits : final + width] = 0
    for label, ids in partitions.items():
        p = predict_all(ids, best)
        results[label] = vocabulary_metrics(p, targets[ids], vocabulary_bits)
        forced_z, diagonal_ablated = [], []
        for start in range(0, len(ids), 128):
            block = ids[start : start + 128]
            state = evolved(block, best[:n_ansatz])
            forced_z.append(sim.probabilities(state))
            ablated_state = evolved(block, no_last_diagonal)
            diagonal_ablated.append(
                sim.readout_probabilities(ablated_state, method, best[n_ansatz:])
            )
        forced_z = np.concatenate(forced_z)
        diagonal_ablated = np.concatenate(diagonal_ablated)
        diagnostics[label] = {
            "readout_reset_bce_delta": binary_cross_entropy(forced_z, target_bits[ids])
            - results[label]["bce"],
            "last_diagonal_ablation_bce_delta": binary_cross_entropy(
                diagonal_ablated, target_bits[ids]
            )
            - results[label]["bce"],
            "last_diagonal_ablation_max_probability_change": float(
                np.max(np.abs(diagonal_ablated - p))
            ),
        }
        payload[f"{label}_ids"] = ids
        payload[f"{label}_probabilities"] = p
        payload[f"{label}_targets"] = targets[ids]
        payload[f"{label}_sentence_ids"] = np.array([examples[i].sentence for i in ids])
    record = {
        "method": method,
        "seed": seed,
        "selected_epoch": best_epoch,
        "parameters": len(best),
        "ansatz_parameters": n_ansatz,
        "readout_parameters": n_readout,
        "initial_validation": initial_val,
        "metrics": results,
        "diagnostics": diagnostics,
        "readout": readout_summary(method, best[n_ansatz:], qubits),
        "history": history,
        "seconds": time.perf_counter() - start_time,
    }
    np.savez_compressed(output / f"{method}_seed{seed}.npz", **payload)
    write_json(result_path, record)
    return {
        "method": method,
        "seed": seed,
        "selected_epoch": best_epoch,
        "seconds": record["seconds"],
        "validation_bce": best_loss,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/measurement_basis/results/confirmation")
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--perturbation", type=float, default=0.1)
    parser.add_argument("--train-limit", type=int, default=0, help="0 means all training examples")
    parser.add_argument("--val-limit", type=int, default=0, help="0 means all validation examples")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(
        args.epochs, args.layers, args.window, args.batch_size, args.eval_every, args.workers
    ) < 1:
        parser.error(
            "Epochs, layers, window, batch size, eval interval and workers must be positive"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    sentences, vocabulary, examples, train, val, test = prepare(args)
    config = vars(args).copy()
    config["data"], config["output"] = str(args.data.resolve()), str(args.output.resolve())
    config.update(
        {
            "objective": "causal",
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "counts": {
                "sentences": len(sentences),
                "vocabulary": len(vocabulary),
                "examples": len(examples),
                "train": len(train),
                "validation": len(val),
                "test": len(test),
            },
            "train_ids": train.tolist(),
            "validation_ids": val.tolist(),
            "test_ids": test.tolist(),
            "optimization": (
                "shared SPSA ansatz gradient and analytic local-readout gradient, mini-batch Adam"
            ),
            "initialization": "matched ansatz per seed; learned readout angles start at zero (Z)",
        }
    )
    existing = args.output / "config.json"
    if existing.exists() and not args.overwrite:
        old = json.loads(existing.read_text())
        ignored = {"workers", "overwrite", "git_head"}
        if {key: value for key, value in old.items() if key not in ignored} != {
            key: value for key, value in config.items() if key not in ignored
        }:
            raise ValueError(
                "Output has a different experiment configuration; choose a new directory"
            )
    write_json(existing, config)
    print(json.dumps({"counts": config["counts"], "jobs": len(args.methods) * len(args.seeds)}))
    jobs = [(vars(args), method, seed) for method in args.methods for seed in args.seeds]
    if args.workers == 1:
        for job in jobs:
            print(json.dumps(fit_one(job)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(fit_one, jobs):
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
