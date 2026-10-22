"""Run one sentence-grouped causal configuration and one repetition."""

import argparse
import time
from pathlib import Path

from qcse.context import ContextConfig
from qcse.data import load_corpus, make_examples, split_examples, word_bits
from qcse.evaluation import evaluation_sample
from qcse.model import QCSEModel
from qcse.outputs import new_run_directory, save_npz, write_json
from qcse.state_cache import ContextStates
from qcse.training import TrainConfig, metrics, train, validation_selection_key


def run(args):
    corpus = load_corpus(
        datasets=("phrases", "cleaned"),
        cleaning="dedupe",
        sampling="uniform",
        seed=args.seed,
    )
    sentences, vocabulary = corpus.sentences, corpus.vocabulary
    examples = make_examples(sentences, vocabulary, window=args.window, objective="causal")
    development_ids, test_ids = split_examples(examples, sentences, args.seed, 0.2)
    development_examples = [examples[index] for index in development_ids]
    train_local, validation_local = split_examples(
        development_examples, sentences, args.seed + 1, 0.2
    )
    train_ids = development_ids[train_local]
    validation_ids = development_ids[validation_local]
    context = ContextConfig(alpha=args.alpha)
    model = QCSEModel(
        vocabulary,
        layers=args.layers,
        context=context,
        window=args.window,
        objective="causal",
        seed=args.seed,
        device=args.device,
        simulation_batch_size=args.simulation_batch_size,
    )
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        samples_per_epoch=args.samples_per_epoch,
        eval_examples=args.eval_examples,
    )
    runtime = {
        "device": args.device,
        "simulation_batch_size": args.simulation_batch_size,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "min_epochs": args.min_epochs,
        "selection_metric": args.selection_metric,
        "seed": args.seed,
    }
    label = f"causal-{args.name}-rep-{args.repetition}"
    with new_run_directory(args.output, label, runtime) as output:
        progress_log = (output / "progress.log").open("w", encoding="utf-8")

        def log(message):
            print(message, flush=True)
            print(message, file=progress_log, flush=True)

        started = time.perf_counter()

        def log_epoch(row):
            validation = row["validation"]
            log(
                f"[{args.name} rep={args.repetition}] "
                f"epoch={row['epoch']:02d} "
                f"train_top1={row['train']['exact_word_accuracy']:.6f} "
                f"val_top1={validation['exact_word_accuracy']:.6f} "
                f"val_similarity={validation['paper_similarity_accuracy']:.6f} "
                f"elapsed_s={time.perf_counter() - started:.1f}"
            )

        log(
            f"[{args.name} rep={args.repetition}] started "
            f"epochs={args.epochs} train_draws={args.samples_per_epoch} "
            f"val_examples={args.eval_examples} test_examples={args.final_eval_examples}"
        )
        provenance = {
            "protocol": "sentence-grouped 64/16/20; one fit; validation-selected epoch",
            "configuration": {
                "name": args.name,
                "alpha": args.alpha,
                "layers": args.layers,
                "window": args.window,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
            },
            "repetition": args.repetition,
            "seed": args.seed,
            "corpus": corpus.summary,
            "sentences": len(sentences),
            "examples": len(examples),
            "development_examples": len(development_ids),
            "train_examples": len(train_ids),
            "validation_examples": len(validation_ids),
            "test_examples": len(test_ids),
            "samples_per_epoch": args.samples_per_epoch,
            "eval_examples": args.eval_examples,
            "final_eval_examples": args.final_eval_examples,
            "test_policy": "held-out test is scored once after validation selection",
        }
        write_json(output / "summary.json", provenance)
        save_npz(
            output / "splits.npz",
            train_ids=train_ids,
            validation_ids=validation_ids,
            test_ids=test_ids,
            development_ids=development_ids,
        )
        history, _ = train(
            model,
            examples,
            train_ids,
            test_ids,
            config,
            validation_ids=validation_ids,
            evaluate_test=False,
            state_cache_mib=args.state_cache_mib,
            early_stopping_patience=args.early_stopping_patience,
            selection_metric=args.selection_metric,
            min_epochs=args.min_epochs,
            callback=log_epoch,
        )
        best = max(history, key=lambda row: validation_selection_key(row, args.selection_metric))
        test_eval_ids = evaluation_sample(test_ids, args.final_eval_examples, args.seed, 2000)
        test_predictions = ContextStates(
            model, [examples[index] for index in test_eval_ids], args.state_cache_mib
        ).predict()
        test_metrics = metrics(
            test_predictions,
            word_bits([examples[index].target for index in test_eval_ids], model.qubits),
        )
        test = {
            **test_metrics,
            "epoch": best["epoch"],
            "selected_by": args.selection_metric,
            "examples": len(test_eval_ids),
            "source_examples": len(test_ids),
            "evaluation_sampling": "fixed seeded sample without replacement",
        }
        model.save(output / "model.npz")
        write_json(output / "history.json", history)
        write_json(
            output / "test.json",
            test
            | {
                "early_stopping": {
                    "patience": args.early_stopping_patience,
                    "min_epochs": args.min_epochs,
                    "stopped_epoch": history[-1]["epoch"],
                }
            },
        )
        log(
            f"Saved {output} at validation-selected epoch {best['epoch']} "
            f"(stopped at {history[-1]['epoch']})",
        )
        progress_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument(
        "--repetition",
        type=int,
        required=True,
        help="Nonnegative repetition index; the caller supplies the corresponding seed",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples-per-epoch", type=int, default=5000)
    parser.add_argument("--eval-examples", type=int, default=2048)
    parser.add_argument("--final-eval-examples", type=int, default=10000)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument(
        "--selection-metric",
        choices=("exact_word_accuracy", "paper_similarity_accuracy", "bce"),
        default="exact_word_accuracy",
    )
    parser.add_argument("--state-cache-mib", type=int, default=256)
    parser.add_argument("--simulation-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/focused-sweep"))
    args = parser.parse_args()
    if (
        min(
            args.alpha,
            args.layers,
            args.window,
            args.batch_size,
            args.learning_rate,
            args.epochs,
            args.samples_per_epoch,
            args.eval_examples,
            args.state_cache_mib,
            args.simulation_batch_size,
        )
        <= 0
    ):
        parser.error("model and training values must be positive")
    if args.final_eval_examples < 0:
        parser.error("final-eval-examples must be nonnegative")
    if args.repetition < 0:
        parser.error("repetition must be nonnegative")
    if args.early_stopping_patience < 1 or args.min_epochs < 0:
        parser.error("early-stopping-patience must be positive and min-epochs nonnegative")
    run(args)


if __name__ == "__main__":
    main()
