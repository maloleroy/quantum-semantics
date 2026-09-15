"""Prepare the corpus, train QCSE, or embed a phrase with a saved model."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from qiskit import qasm3

from .circuit import DEFAULT_LAYERS
from .context import METHODS, ContextConfig, context_matrix
from .cross_validation import cross_validate
from .data import DATASETS, DEFAULT_DATA, load_corpus, load_phrases, make_examples, split_examples
from .model import QCSEModel
from .outputs import new_run_directory, run_status, save_npz, write_json
from .training import TrainConfig, load_run, save_run, train


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    execution = argparse.ArgumentParser(add_help=False)
    execution.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    execution.add_argument(
        "--state-cache-mib",
        type=int,
        default=256,
        help="Budget for retained encoded contexts; larger corpora stream in batches",
    )
    execution.add_argument(
        "--simulation-batch-size",
        type=int,
        default=256,
        help="Contexts simulated in parallel per weight vector (independent of Adam batch size)",
    )
    sub = root.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train"):
        p = sub.add_parser(command, parents=[execution])
        data = p.add_mutually_exclusive_group()
        data.add_argument("--data", type=Path, nargs="+", help="Custom sentence files")
        data.add_argument(
            "--datasets",
            choices=tuple(DATASETS),
            nargs="+",
            help="Training sources (default: both); vocabulary uses phrases and cleaned",
        )
        p.add_argument("--cleaning", choices=("basic", "dedupe", "strict"), default="dedupe")
        p.add_argument("--sampling", choices=("uniform", "balanced"), default="uniform")
        p.add_argument(
            "--max-sentences", type=int, help="Sample sentences, keeping the full vocabulary"
        )
        p.add_argument(
            "--output",
            type=Path,
            default=DEFAULT_DATA.parent / "outputs",
            help="Parent directory; every invocation creates a unique run subfolder",
        )
        p.add_argument("--method", choices=METHODS, default="exponential")
        p.add_argument("--alpha", type=float, default=1.0)
        p.add_argument("--omega", type=float, default=1.0)
        p.add_argument("--delta", type=float, default=1.0)
        p.add_argument("--prime", type=int, default=31)
        p.add_argument("--hash-size", type=int, default=997)
        p.add_argument(
            "--window", type=int, default=4, help="CBOW total context, or causal prefix length"
        )
        p.add_argument(
            "--objective",
            choices=("causal", "cbow"),
            default="causal",
            help="Training and inference objective (causal is GPT-like; cbow is BERT-like)",
        )
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
            p.add_argument(
                "--samples-per-epoch",
                type=int,
                help="Training examples sampled with replacement per epoch (default: full passes)",
            )
            p.add_argument(
                "--eval-examples",
                type=int,
                default=2048,
                help="Fixed monitoring sample per split for sampled epochs (default: 2048)",
            )
            p.add_argument(
                "--final-eval-examples",
                type=int,
                default=10000,
                help="Final validation/test examples per split (0=all; default: 10000)",
            )
            p.add_argument("--batch-size", type=int, default=32)
            p.add_argument("--learning-rate", type=float, default=0.0003)
            p.add_argument("--l2", type=float, default=0.001)
            p.add_argument("--perturbation", type=float, default=0.1)
            p.add_argument("--test-fraction", type=float, default=0.2)
            p.add_argument(
                "--folds",
                type=int,
                default=1,
                help="Use 2+ for cross-validation within the 80%% development split",
            )
            p.add_argument(
                "--val-fraction",
                type=float,
                help="Validation fraction of dev for shuffle-split CV (default: 1/folds k-fold)",
            )
            p.add_argument(
                "--max-examples",
                type=int,
                help="Random subset AFTER sentence split; full vocabulary retained",
            )
    p = sub.add_parser("embed", parents=[execution])
    p.add_argument("phrase")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--max-new-tokens", type=int, default=0)
    p = sub.add_parser(
        "continue", parents=[execution], help="Resume a saved run for additional epochs"
    )
    p.add_argument("run", type=Path, help="Run archive, e.g. outputs/train/run.npz")
    p.add_argument("--epochs", type=int, required=True, help="Additional epochs to train")
    p.add_argument(
        "--output",
        type=Path,
        help="Parent for a new continuation folder (default: update the existing run)",
    )
    p = sub.add_parser(
        "resume-cv",
        parents=[execution],
        help="Resume unfinished folds/refit in an existing cross-validation directory",
    )
    p.add_argument("run", type=Path)
    return root


def _write_training_outputs(
    output, model, embeddings, examples, original_ids, train_ids, test_ids, example
):
    model.save(output / "model.npz")
    if len(embeddings):
        save_npz(
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
    circuit = model.circuit(example.context)
    (output / "circuit.txt").write_text(
        str(circuit.draw(output="text", fold=120)), encoding="utf-8"
    )
    (output / "circuit.qasm").write_text(qasm3.dumps(circuit), encoding="utf-8")


def continue_run(args):
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    saved = load_run(args.run)
    metadata = saved["metadata"]
    model_metadata = metadata["model"]
    model = QCSEModel(
        model_metadata["vocabulary"],
        layers=model_metadata["layers"],
        context=ContextConfig(**model_metadata["context"]),
        window=model_metadata["window"],
        objective=model_metadata.get("objective", "cbow"),
        direction=model_metadata["direction"],
        seed=0,
        device=args.device,
        simulation_batch_size=args.simulation_batch_size,
    )
    model.weights = saved["weights"]
    output = args.output or args.run.parent
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "vocabulary.json", model.vocabulary)
    if metadata.get("provenance"):
        write_json(output / "summary.json", metadata["provenance"])
    examples = saved["examples"]
    train_ids, test_ids = saved["train_ids"], saved["test_ids"]
    original_ids = saved["original_ids"]
    old_config = metadata["config"]
    config = TrainConfig(
        args.epochs,
        old_config["batch_size"],
        old_config["learning_rate"],
        old_config["l2"],
        old_config["perturbation"],
        old_config["seed"],
        old_config.get("samples_per_epoch"),
        old_config.get("eval_examples", 2048),
    )
    state = saved["state"]
    history_rows = list(state["history"])
    validation_ids = saved["validation_ids"]
    evaluate_test = metadata.get("evaluation", "test") == "test"

    def progress(row):
        print_progress(row)
        history_rows.append(row)
        write_json(output / "history.json", history_rows)

    def checkpoint(current):
        save_run(
            output / "run.npz",
            model,
            current,
            config=config,
            examples=examples,
            train_ids=train_ids,
            test_ids=test_ids,
            original_ids=original_ids,
            provenance=metadata.get("provenance"),
            validation_ids=validation_ids,
            evaluate_test=evaluate_test,
        )

    _, embeddings = train(
        model,
        examples,
        train_ids,
        test_ids,
        config,
        progress,
        initial_state=state,
        checkpoint_callback=checkpoint,
        validation_ids=validation_ids,
        evaluate_test=evaluate_test,
        state_cache_mib=args.state_cache_mib,
    )
    write_json(output / "training_config.json", old_config | {"epochs_added": args.epochs})
    _write_training_outputs(
        output, model, embeddings, examples, original_ids, train_ids, test_ids, examples[0]
    )
    print(f"Continued run through epoch {history_rows[-1]['epoch']} and saved it to {output}")


def print_progress(row):
    scope = " (sample)" if "metric_examples" in row else ""
    text = f"Epoch {row['epoch']:3d}{scope}: train BCE={row['train']['bce']:.8f}"
    for name in ("validation", "test"):
        if name in row:
            text += f", {name} BCE={row[name]['bce']:.8f}"
            text += f", exact word={row[name]['exact_word_accuracy']:.6f}"
    print(text, flush=True)


def resume_cv(args):
    output = args.run
    summary = json.loads((output / "summary.json").read_text())
    settings = json.loads((output / "training_config.json").read_text())
    vocabulary = json.loads((output / "vocabulary.json").read_text())
    sentences = load_phrases(output / "sentences.csv")
    examples = make_examples(sentences, vocabulary, summary["window"], summary["objective"])
    with np.load(output / "splits.npz", allow_pickle=False) as split:
        original_ids = split["original_example_ids"].copy()
        development_ids, test_ids = split["development_ids"].copy(), split["test_ids"].copy()
    examples = [examples[i] for i in original_ids]
    config = TrainConfig(
        **{key: settings[key] for key in TrainConfig.__dataclass_fields__ if key in settings}
    )
    model = QCSEModel(
        vocabulary,
        layers=summary["ansatz_layers"],
        context=ContextConfig(**summary["context"]),
        window=summary["window"],
        objective=summary["objective"],
        direction=summary["direction"],
        seed=summary["seed"],
        device=args.device,
        simulation_batch_size=args.simulation_batch_size,
    )
    return cross_validate(
        model,
        examples,
        sentences,
        development_ids,
        test_ids,
        config,
        output,
        original_ids,
        summary,
        settings["folds"],
        args.state_cache_mib,
        settings.get("val_fraction"),
        settings.get("final_eval_examples", 10000),
    )


def run(args):
    if args.command == "embed":
        model = QCSEModel.load(
            args.model, device=args.device, simulation_batch_size=args.simulation_batch_size
        )
        result = (
            model.complete_phrase(args.phrase, args.max_new_tokens)
            if args.max_new_tokens
            else model.embed_phrase(args.phrase)
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    if args.command == "continue":
        if args.output:
            with new_run_directory(args.output, "continue", runtime_info(args)) as output:
                args.output = output
                return continue_run(args)
        with run_status(args.run.parent, runtime_info(args)):
            return continue_run(args)
    if args.command == "resume-cv":
        with run_status(args.run, runtime_info(args)):
            return resume_cv(args)
    label = f"{args.command}-{args.objective}-l{args.layers}"
    with new_run_directory(args.output, label, runtime_info(args)) as output:
        return run_corpus(args, output)


def runtime_info(args):
    return {
        "device": args.device,
        "simulation_batch_size": args.simulation_batch_size,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "arguments": {
            key: str(value)
            if isinstance(value, Path)
            else [str(p) for p in value]
            if isinstance(value, list)
            else value
            for key, value in vars(args).items()
        },
    }


def run_corpus(args, output):
    if args.command == "train":
        if args.folds < 1:
            raise ValueError("folds must be positive")
        if args.folds > 1 and args.max_examples is not None:
            raise ValueError(
                "Use max-sentences with cross-validation to retain whole sentence groups"
            )
        if args.eval_examples < 1 or args.final_eval_examples < 0:
            raise ValueError("eval-examples must be positive and final-eval-examples nonnegative")
    corpus = load_corpus(
        paths=args.data,
        datasets=args.datasets,
        cleaning=args.cleaning,
        sampling=args.sampling,
        max_sentences=args.max_sentences,
        seed=args.seed,
    )
    sentences, vocabulary = corpus.sentences, corpus.vocabulary
    examples = make_examples(sentences, vocabulary, args.window, args.objective)
    if not examples:
        raise ValueError("No phrases with at least two words")
    context = ContextConfig(
        args.method, args.alpha, args.omega, args.delta, args.prime, args.hash_size
    )
    model = QCSEModel(
        vocabulary,
        layers=args.layers,
        context=context,
        window=args.window,
        objective=args.objective,
        direction=args.direction,
        seed=args.seed,
        device=args.device,
        simulation_batch_size=args.simulation_batch_size,
    )
    summary = {
        **corpus.summary,
        "sentences": len(sentences),
        "tokens": sum(map(len, sentences)),
        "vocabulary_size": len(vocabulary),
        "qubits": model.qubits,
        "examples": len(examples),
        "ansatz_layers": model.layers,
        "trainable_parameters": len(model.weights),
        "context": asdict(context),
        "window": args.window,
        "objective": args.objective,
        "direction": args.direction,
        "seed": args.seed,
        "bit_order": "q0 first (least significant bit)",
        "vocabulary_order": "descending corpus frequency, alphabetical ties",
    }
    write_json(output / "vocabulary.json", vocabulary)
    write_json(output / "summary.json", summary)
    write_json(output / "sentence_sources.json", corpus.sources)
    with (output / "sentences.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["sentence"])
        writer.writerows([" ".join(words)] for words in sentences)
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
        save_npz(
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
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.l2,
        args.perturbation,
        args.seed,
        args.samples_per_epoch,
        args.eval_examples,
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
            "folds": args.folds,
            "val_fraction": args.val_fraction,
            "final_eval_examples": args.final_eval_examples,
        },
    )
    schedule = (
        f"{args.samples_per_epoch} sampled examples per epoch"
        if args.samples_per_epoch
        else "full data passes"
    )
    print(
        f"Training from {len(examples)} eligible examples using {schedule}. "
        "Epoch 0 is the baseline.",
        flush=True,
    )
    if args.folds > 1:
        return cross_validate(
            model,
            examples,
            sentences,
            train_ids,
            test_ids,
            config,
            output,
            original_ids,
            summary,
            args.folds,
            args.state_cache_mib,
            args.val_fraction,
            args.final_eval_examples,
        )

    def progress(row):
        print_progress(row)
        progress.rows.append(row)
        write_json(output / "history.json", progress.rows)

    progress.rows = []

    def checkpoint(state):
        save_run(
            output / "run.npz",
            model,
            state,
            config=config,
            examples=examples,
            train_ids=train_ids,
            test_ids=test_ids,
            original_ids=original_ids,
            provenance=summary,
        )

    _, embeddings = train(
        model,
        examples,
        train_ids,
        test_ids,
        config,
        progress,
        checkpoint_callback=checkpoint,
        state_cache_mib=args.state_cache_mib,
    )
    _write_training_outputs(
        output, model, embeddings, examples, original_ids, train_ids, test_ids, example
    )
    print(f"Saved training artifacts to {output}")


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as error:
        cli.exit(2, f"qcse: {error}\n")
    except KeyboardInterrupt:
        cli.exit(130, "qcse: interrupted; the last completed epoch is saved in run.npz\n")


if __name__ == "__main__":
    main()
