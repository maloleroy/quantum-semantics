"""Prepare the corpus, train QCSE, or embed a phrase with a saved model."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from qiskit import qasm3

from .circuit import DEFAULT_LAYERS
from .context import METHODS, ContextConfig, context_matrix
from .data import DEFAULT_DATA, build_vocabulary, load_phrases, make_examples, split_examples
from .model import QCSEModel
from .training import TrainConfig, train


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train"):
        p = sub.add_parser(command)
        p.add_argument("--data", type=Path, default=DEFAULT_DATA)
        p.add_argument("--output", type=Path, default=DEFAULT_DATA.parent / "outputs" / command)
        p.add_argument("--method", choices=METHODS, default="exponential")
        p.add_argument("--alpha", type=float, default=1.0)
        p.add_argument("--omega", type=float, default=1.0)
        p.add_argument("--delta", type=float, default=1.0)
        p.add_argument("--prime", type=int, default=31)
        p.add_argument("--hash-size", type=int, default=997)
        p.add_argument("--window", type=int, default=4, help="Total context size, half each side")
        p.add_argument(
            "--layers",
            type=int,
            default=DEFAULT_LAYERS,
            help="Number of trainable ansatz layers",
        )
        p.add_argument("--direction", choices=("forward", "reverse"), default="forward")
        p.add_argument("--seed", type=int, default=42)
        if command == "train":
            p.add_argument("--epochs", type=int, default=50)
            p.add_argument("--batch-size", type=int, default=32)
            p.add_argument("--learning-rate", type=float, default=0.0003)
            p.add_argument("--l2", type=float, default=0.001)
            p.add_argument("--perturbation", type=float, default=0.1)
            p.add_argument("--test-fraction", type=float, default=0.2)
            p.add_argument(
                "--max-examples",
                type=int,
                help="Random subset AFTER sentence split; full vocabulary retained",
            )
    p = sub.add_parser("embed")
    p.add_argument("phrase")
    p.add_argument("--model", type=Path, default=DEFAULT_DATA.parent / "outputs/train/model.npz")
    p.add_argument("--output", type=Path)
    return root


def run(args):
    if args.command == "embed":
        result = QCSEModel.load(args.model).embed_phrase(args.phrase)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    sentences = load_phrases(args.data)
    vocabulary = build_vocabulary(sentences)
    examples = make_examples(sentences, vocabulary, args.window)
    if not examples:
        raise ValueError("No phrases with at least two words")
    context = ContextConfig(
        args.method, args.alpha, args.omega, args.delta, args.prime, args.hash_size
    )
    model = QCSEModel(
        vocabulary, layers=args.layers, context=context, window=args.window,
        direction=args.direction, seed=args.seed
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": str(args.data.resolve()),
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "sentences": len(sentences),
        "tokens": sum(map(len, sentences)),
        "vocabulary_size": len(vocabulary),
        "qubits": model.qubits,
        "examples": len(examples),
        "ansatz_layers": model.layers,
        "trainable_parameters": len(model.weights),
        "context": asdict(context),
        "window": args.window,
        "direction": args.direction,
        "seed": args.seed,
        "bit_order": "q0 first (least significant bit)",
        "vocabulary_order": "descending corpus frequency, alphabetical ties",
    }
    write_json(output / "vocabulary.json", vocabulary)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    example = examples[0]
    matrix = context_matrix(example.context, len(vocabulary), context)
    angles = model.angles(example.context)
    write_json(
        output / "example.json",
        {
            "sentence": " ".join(sentences[example.sentence]),
            "position": example.position,
            "center": vocabulary[example.target],
            "context": [vocabulary[i] for i in example.context],
            "matrix": matrix.tolist(),
            "angles_L_m_2": angles.tolist(),
            "reshaped_2m_L": angles.reshape(len(angles), -1).T.tolist(),
        },
    )
    circuit = model.circuit(example.context)
    (output / "circuit.txt").write_text(
        str(circuit.draw(output="text", fold=120)), encoding="utf-8"
    )
    (output / "circuit.qasm").write_text(qasm3.dumps(circuit), encoding="utf-8")
    if args.command == "prepare":
        # Variable-size matrices are stored with offsets, without pickle/object arrays.
        matrices = [context_matrix(e.context, len(vocabulary), context) for e in examples]
        sizes = np.array([m.size for m in matrices])
        context_sizes = np.array([len(e.context) for e in examples])
        np.savez_compressed(
            output / "contexts.npz",
            matrix_values=np.concatenate([m.ravel() for m in matrices]),
            matrix_offsets=np.r_[0, sizes.cumsum()],
            matrix_shapes=np.array([m.shape for m in matrices]),
            context_ids=np.concatenate([e.context for e in examples]),
            context_offsets=np.r_[0, context_sizes.cumsum()],
            target_ids=[e.target for e in examples],
            sentence_ids=[e.sentence for e in examples],
            positions=[e.position for e in examples],
        )
        print(f"Prepared every context matrix in {output}")
        return
    train_ids, test_ids = split_examples(examples, sentences, args.seed, args.test_fraction)
    original_ids = np.arange(len(examples))
    if args.max_examples is not None:
        if args.max_examples < 2:
            raise ValueError("max-examples must be at least 2")
        if args.max_examples < len(examples):
            rng = np.random.default_rng(args.seed)
            ntest = max(1, min(len(test_ids), round(args.max_examples * args.test_fraction)))
            ntrain = min(len(train_ids), args.max_examples - ntest)
            if ntrain < 1:
                raise ValueError("Subset must contain a training example")
            original_ids = np.r_[
                rng.choice(train_ids, ntrain, replace=False),
                rng.choice(test_ids, ntest, replace=False),
            ]
            examples = [examples[i] for i in original_ids]
            train_ids = np.arange(ntrain)
            test_ids = np.arange(ntrain, len(examples))
    config = TrainConfig(
        args.epochs, args.batch_size, args.learning_rate, args.l2, args.perturbation, args.seed
    )
    write_json(
        output / "training_config.json",
        asdict(config)
        | {
            "optimizer": "Adam with SPSA gradients",
            "test_fraction": args.test_fraction,
            "max_examples": args.max_examples,
            "train_examples": len(train_ids),
            "test_examples": len(test_ids),
            "split": "grouped by sentence text",
        },
    )
    print(f"Encoding {len(examples)} examples, then training. Epoch 0 is the baseline.", flush=True)

    def progress(row):
        print(
            f"Epoch {row['epoch']:3d}: train BCE={row['train']['bce']:.6f}, "
            f"test BCE={row['test']['bce']:.6f}, "
            f"exact word={row['test']['exact_word_accuracy']:.3f}, "
            f"paper score={row['test']['paper_similarity_accuracy']:.3f}",
            flush=True,
        )
        write_json(output / "history.json", history_rows := progress.rows + [row])
        progress.rows = history_rows

    progress.rows = []
    _, embeddings = train(model, examples, train_ids, test_ids, config, progress)
    model.save(output / "model.npz")
    np.savez_compressed(
        output / "embeddings.npz",
        probabilities=embeddings,
        pauli_z=1 - 2 * embeddings,
        target_ids=[e.target for e in examples],
        sentence_ids=[e.sentence for e in examples],
        positions=[e.position for e in examples],
        original_example_ids=original_ids,
        train_ids=train_ids,
        test_ids=test_ids,
    )
    # Export the trained circuit, replacing the initialized preview for training runs.
    circuit = model.circuit(example.context)
    (output / "circuit.txt").write_text(
        str(circuit.draw(output="text", fold=120)), encoding="utf-8"
    )
    (output / "circuit.qasm").write_text(qasm3.dumps(circuit), encoding="utf-8")
    print(f"Saved model, contextual embeddings, circuit and metrics to {output}")


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as error:
        cli.exit(2, f"qcse: {error}\n")


if __name__ == "__main__":
    main()
