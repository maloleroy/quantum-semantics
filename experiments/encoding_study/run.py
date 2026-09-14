"""Run from QCSE: uv run python -m experiments.encoding_study.run --help."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from qcse.data import DEFAULT_DATA, build_vocabulary, load_phrases, make_examples, split_examples
from qcse.training import binary_cross_entropy, metrics

from .encoders import METHODS, Features
from .simulator import Simulator


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def vocabulary_metrics(p, targets, bits):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    log_scores = np.log(p) @ bits.T + np.log1p(-p) @ (1 - bits).T
    row_max = log_scores.max(axis=1, keepdims=True)
    log_z = row_max[:, 0] + np.log(np.exp(log_scores - row_max).sum(axis=1))
    ranks = np.argsort(-log_scores, axis=1, kind="stable")
    result = metrics(p, bits[targets])
    result.update(
        {
            "vocab_top1": float(np.mean(ranks[:, 0] == targets)),
            "vocab_top5": float(np.mean(np.any(ranks[:, :5] == targets[:, None], axis=1))),
            "vocab_nll": float(np.mean(log_z - log_scores[np.arange(len(targets)), targets])),
            "invalid_id_rate": float(
                np.mean((p >= 0.5) @ (1 << np.arange(p.shape[1])) >= len(bits))
            ),
        }
    )
    return result


def prepare(args):
    sentences = load_phrases(args.data)
    vocabulary = build_vocabulary(sentences)
    examples = make_examples(sentences, vocabulary, 4, "causal")
    outer_train, test = split_examples(examples, sentences, seed=42)
    # A second sentence-group split inside outer_train; no shared sentences with validation.
    inside = [examples[i] for i in outer_train]
    local_train, local_val = split_examples(inside, sentences, seed=31415, test_fraction=0.2)
    train, val = outer_train[local_train], outer_train[local_val]
    rng = np.random.default_rng(1618)
    if args.train_limit:
        train = np.sort(rng.choice(train, min(len(train), args.train_limit), replace=False))
    if args.val_limit:
        val = np.sort(rng.choice(val, min(len(val), args.val_limit), replace=False))
    return vocabulary, examples, train, val, test


def fit_one(job):
    args_dict, method, seed = job
    args = argparse.Namespace(**args_dict)
    output = Path(args.output)
    result_path = output / f"{method}_seed{seed}.json"
    if result_path.exists() and not args.overwrite:
        return {"method": method, "seed": seed, "cached": True}
    start = time.perf_counter()
    vocabulary, examples, train, val, test = prepare(args)
    qubits = (len(vocabulary) - 1).bit_length()
    if qubits != 10:
        raise ValueError(
            "This study's positional feature layout requires 10 qubits (513-1024 words)"
        )
    features = Features(examples, train, len(vocabulary), qubits)
    spec = features.build(method)
    sim = Simulator(qubits)
    n_ansatz = args.layers * (3 * qubits - 1)
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-np.pi, np.pi, n_ansatz)
    if spec["dynamic"] == "affine":
        weights = np.r_[weights, np.ones(2 * qubits), np.zeros(2 * qubits)]
    elif spec["dynamic"] == "table":
        weights = np.r_[weights, features.random.ravel()]
    angles = spec["angles"]
    states = spec["state"]
    if states is None and not spec["dynamic"]:
        states = np.concatenate(
            [
                sim.encode(angles[lo : lo + 128], spec["kind"], spec["entangle"])
                for lo in range(0, len(examples), 128)
            ]
        )
    targets = np.array([e.target for e in examples])
    target_bits = features.target_bits[targets]

    def predict(ids, w):
        if spec["dynamic"]:
            if spec["dynamic"] == "affine":
                scale, offset = w[n_ansatz:].reshape(2, -1)
                x = angles[ids].reshape(len(ids), -1) * scale + offset
                a = x.reshape(len(ids), 1, qubits, 2)
            else:
                table = w[n_ansatz:].reshape(len(vocabulary), 2 * qubits)
                a = []
                for idx in ids:
                    ctx = examples[idx].context
                    recency = 0.5 ** np.arange(len(ctx))[::-1]
                    a.append(recency @ table[list(ctx)] / recency.sum())
                a = np.asarray(a).reshape(len(ids), 1, qubits, 2)
            state = sim.encode(a, spec["kind"], spec["entangle"])
        else:
            state = states[ids]
        reupload = angles[ids] if spec["reupload"] else None
        return sim.probabilities(sim.ansatz(state, w[:n_ansatz], reupload))

    def predict_all(ids, w):
        return np.concatenate([predict(ids[i : i + 128], w) for i in range(0, len(ids), 128)])

    def penalty_gradient(w):
        # Match repository L2 on ansatz. Affine/table adaptation has no additional L2.
        g = np.zeros_like(w)
        g[:n_ansatz] = 2 * args.l2 * w[:n_ansatz]
        return g

    initial_val = vocabulary_metrics(predict_all(val, weights), targets[val], features.target_bits)
    history = [{"epoch": 0, "val_bce": initial_val["bce"]}]
    best_loss, best_epoch, best = initial_val["bce"], 0, weights.copy()
    first, second, step = np.zeros_like(weights), np.zeros_like(weights), 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train)
        for lo in range(0, len(order), args.batch_size):
            ids = order[lo : lo + args.batch_size]
            step += 1
            delta = rng.choice([-1.0, 1.0], size=len(weights))
            c = 0.1 / step**0.101
            plus = binary_cross_entropy(predict(ids, weights + c * delta), target_bits[ids])
            minus = binary_cross_entropy(predict(ids, weights - c * delta), target_bits[ids])
            gradient = (plus - minus) / (2 * c) * delta + penalty_gradient(weights)
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            weights -= (
                args.learning_rate
                * first
                / (1 - 0.9**step)
                / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            loss = binary_cross_entropy(predict_all(val, weights), target_bits[val])
            history.append({"epoch": epoch, "val_bce": loss})
            if loss < best_loss:
                best_loss, best_epoch, best = loss, epoch, weights.copy()
    partitions = {"train": train, "validation": val}
    if args.test:
        partitions["test"] = test
    results = {}
    payload = {"weights": best, "vocabulary": np.array(vocabulary)}
    for label, ids in partitions.items():
        p = predict_all(ids, best)
        results[label] = vocabulary_metrics(p, targets[ids], features.target_bits)
        payload[f"{label}_ids"] = ids
        payload[f"{label}_probabilities"] = p
        payload[f"{label}_targets"] = targets[ids]
        payload[f"{label}_sentence_ids"] = np.array([examples[i].sentence for i in ids])
    record = {
        "method": method,
        "seed": seed,
        "selected_epoch": best_epoch,
        "parameters": len(weights),
        "ansatz_parameters": n_ansatz,
        "initial_validation": initial_val,
        "metrics": results,
        "history": history,
        "seconds": time.perf_counter() - start,
    }
    np.savez_compressed(output / f"{method}_seed{seed}.npz", **payload)
    write_json(result_path, record)
    return {k: record[k] for k in ("method", "seed", "selected_epoch", "seconds")} | {
        "val_bce": best_loss,
    }


def baseline_metrics(args):
    vocabulary, examples, train, val, test = prepare(args)
    qubits = (len(vocabulary) - 1).bit_length()
    features = Features(examples, train, len(vocabulary), qubits)
    result = {}
    partitions = {"validation": val} | ({"test": test} if args.test else {})
    for label, ids in partitions.items():
        targets = np.array([examples[i].target for i in ids])
        last = np.array([examples[i].context[-1] for i in ids])
        rows = {}
        for name, p in {
            "bit_prior": np.broadcast_to(features.bit_prior, (len(ids), qubits)),
            "bigram_bits": features.bigram_bits[last],
            "always_id_zero": np.full((len(ids), qubits), 1e-10),
        }.items():
            rows[name] = vocabulary_metrics(p, targets, features.target_bits)
        for name, p in {
            "unigram": np.broadcast_to(features.unigram, (len(ids), len(vocabulary))),
            "bigram": features.bigram[last],
        }.items():
            ranking = np.argsort(-p, axis=1, kind="stable")
            rows[name] = {
                "vocab_nll": float(-np.log(p[np.arange(len(ids)), targets]).mean()),
                "vocab_top1": float(np.mean(ranking[:, 0] == targets)),
                "vocab_top5": float(np.mean(np.any(ranking[:, :5] == targets[:, None], axis=1))),
            }
        result[label] = rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/encoding_study/results/screen")
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=[method for method in METHODS if method != "constant_zero"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument(
        "--train-limit", type=int, default=640, help="0 means all training examples"
    )
    parser.add_argument(
        "--val-limit", type=int, default=320, help="0 means all validation examples"
    )
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--test", action="store_true", help="Only for preselected confirmation runs"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.layers, args.batch_size, args.eval_every, args.workers) < 1:
        parser.error("Epochs, layers, batch size, eval interval and workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    vocabulary, examples, train, val, test = prepare(args)
    config = vars(args).copy()
    config["data"], config["output"] = str(args.data.resolve()), str(args.output.resolve())
    config.update(
        {
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "counts": {
                "vocabulary": len(vocabulary),
                "examples": len(examples),
                "train": len(train),
                "validation": len(val),
                "test": len(test),
            },
            "train_ids": train.tolist(),
            "validation_ids": val.tolist(),
            "test_ids": test.tolist(),
        }
    )
    # Resume only an identical design; a new experiment needs a new output directory.
    existing = args.output / "config.json"
    if existing.exists() and not args.overwrite:
        old = json.loads(existing.read_text())
        ignored = {"workers", "overwrite", "git_head"}
        if {k: v for k, v in old.items() if k not in ignored} != {
            k: v for k, v in config.items() if k not in ignored
        }:
            raise ValueError(
                "Output has a different experiment configuration; choose a new directory"
            )
    write_json(existing, config)
    write_json(args.output / "baselines.json", baseline_metrics(args))
    print(
        json.dumps({"counts": config["counts"], "jobs": len(args.methods) * len(args.seeds)}),
        flush=True,
    )
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
