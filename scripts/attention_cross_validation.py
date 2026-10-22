"""Run grouped cross-validation for the two quantum-attention input encoders."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from qcse.attention import QuantumAttentionModel
from qcse.data import load_corpus, make_examples, split_examples, validation_folds
from qcse.evaluation import evaluation_sample
from qcse.outputs import write_json

SEEDS = (42, 43)
SETUPS = (
    {"name": "qcse-1layer", "encoding": "qcse", "layers": 1, "circuit": True},
    {"name": "classical-1layer", "encoding": "classical", "layers": 1, "circuit": True},
    {"name": "qcse-2layer", "encoding": "qcse", "layers": 2, "circuit": True},
    {"name": "classical-2layer", "encoding": "classical", "layers": 2, "circuit": True},
    {"name": "classical-no-circuit", "encoding": "classical", "layers": 1, "circuit": False},
)
HYPERPARAMETER_SETUPS = (
    {
        "name": "qcse-reference",
        "encoding": "qcse",
        "layers": 1,
        "circuit": True,
        "alpha": 0.05,
        "learning_rate": 0.003,
    },
    {
        "name": "classical-lr001",
        "encoding": "classical",
        "layers": 1,
        "circuit": True,
        "alpha": 0.05,
        "learning_rate": 0.001,
    },
    {
        "name": "classical-lr003",
        "encoding": "classical",
        "layers": 1,
        "circuit": True,
        "alpha": 0.05,
        "learning_rate": 0.003,
    },
    {
        "name": "classical-lr01",
        "encoding": "classical",
        "layers": 1,
        "circuit": True,
        "alpha": 0.05,
        "learning_rate": 0.01,
    },
    {
        "name": "classical-alpha001",
        "encoding": "classical",
        "layers": 1,
        "circuit": True,
        "alpha": 0.01,
        "learning_rate": 0.003,
    },
    {
        "name": "classical-alpha02",
        "encoding": "classical",
        "layers": 1,
        "circuit": True,
        "alpha": 0.2,
        "learning_rate": 0.003,
    },
    {
        "name": "classical-2layer",
        "encoding": "classical",
        "layers": 2,
        "circuit": True,
        "alpha": 0.05,
        "learning_rate": 0.003,
    },
    {
        "name": "classical-no-circuit",
        "encoding": "classical",
        "layers": 1,
        "circuit": False,
        "alpha": 0.05,
        "learning_rate": 0.003,
    },
)


def best_validation_row(history):
    """Select an epoch by retrieval top-1, with deterministic secondary keys."""
    return max(
        history,
        key=lambda row: (
            row["validation"]["cosine_top1"],
            row["validation"]["cosine_top5"],
            -row["validation"]["cross_entropy"],
            -row["epoch"],
        ),
    )


@torch.no_grad()
def evaluate(model, contexts, targets, ids, batch_size):
    if len(ids) == 0:
        raise ValueError("Cannot evaluate an empty split")
    cross_entropy = top1 = top5 = 0.0
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        scores = model.scores(contexts[batch])
        target = targets[batch]
        cross_entropy += float(F.cross_entropy(scores, target, reduction="sum"))
        cosine = model.scores(contexts[batch], similarity="cosine")
        top1 += float((cosine.argmax(-1) == target).sum())
        top5 += float(
            (cosine.topk(min(5, cosine.shape[-1]), dim=-1).indices == target[:, None]).any(-1).sum()
        )
    return {
        "cross_entropy": cross_entropy / len(ids),
        "perplexity": math.exp(min(cross_entropy / len(ids), 80)),
        "cosine_top1": top1 / len(ids),
        "cosine_top5": top5 / len(ids),
        "examples": len(ids),
    }


def fit(model, contexts, targets, train_ids, val_ids, evaluation_ids, settings, seed, output):
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    rng = np.random.default_rng(seed)
    monitor_rng = np.random.default_rng(seed + 1000)
    monitor_size = settings["eval_examples"]
    train_monitor = monitor_rng.choice(train_ids, min(monitor_size, len(train_ids)), replace=False)
    val_monitor = evaluation_sample(val_ids, monitor_size, seed, 3000)
    history = []
    for epoch in range(settings["epochs"] + 1):
        begun = time.monotonic()
        train_metrics = evaluate(model, contexts, targets, train_monitor, settings["batch_size"])
        val_metrics = evaluate(model, contexts, targets, val_monitor, settings["batch_size"])
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        if epoch == settings["epochs"]:
            break
        order = rng.choice(train_ids, settings["samples_per_epoch"], replace=True)
        model.train()
        for offset in range(0, len(order), settings["batch_size"]):
            batch = order[offset : offset + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(contexts[batch]), targets[batch])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
        history[-1]["seconds"] = time.monotonic() - begun
        if epoch == settings["epochs"] - 1:
            print(f"{output.name}: completed epoch {settings['epochs']}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "history.json", history)
    torch.save(
        {"model_config": model.config, "model": model.state_dict(), "seed": seed},
        output / "checkpoint.pt",
    )
    full_validation = evaluate(model, contexts, targets, evaluation_ids, settings["batch_size"])
    full_validation["source_examples"] = len(val_ids)
    full_validation["evaluation_sampling"] = (
        "fixed seeded sample without replacement"
        if len(evaluation_ids) < len(val_ids)
        else "complete validation split"
    )
    write_json(output / "full_validation.json", full_validation)
    return history, full_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/attention-cv-50"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples-per-epoch", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--final-eval-examples", type=int, default=10000)
    parser.add_argument("--max-sentences", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--profile", choices=("ablation", "hyperparameters"), default="ablation")
    args = parser.parse_args()
    if (
        min(args.epochs, args.samples_per_epoch, args.batch_size, args.eval_examples, args.threads)
        < 1
    ):
        parser.error("epochs, samples, batch size, eval examples and threads must be positive")
    if args.final_eval_examples < 0:
        parser.error("final-eval-examples must be nonnegative")
    if args.max_sentences == 1 or args.max_sentences < 0:
        parser.error("max-sentences must be zero (full corpus) or at least two")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    corpus = load_corpus(
        datasets=("phrases", "cleaned"),
        sampling="balanced",
        max_sentences=args.max_sentences or None,
        seed=42,
    )
    examples = make_examples(corpus.sentences, corpus.vocabulary, window=4, objective="causal")
    dev_ids, test_ids = split_examples(examples, corpus.sentences, seed=42)
    contexts = torch.full((len(examples), 4), len(corpus.vocabulary), dtype=torch.long)
    for index, example in enumerate(examples):
        contexts[index, -len(example.context) :] = torch.tensor(example.context)
    targets = torch.tensor([example.target for example in examples])
    contexts, targets = contexts.to(args.device), targets.to(args.device)
    settings = vars(args).copy()
    settings["output"] = str(settings["output"])
    settings["datasets"] = ["phrases", "cleaned"]
    settings["sampling"] = "balanced"
    settings["vocabulary_size"] = len(corpus.vocabulary)
    settings["sentences"] = len(corpus.sentences)
    settings["examples"] = len(examples)
    settings["development_examples"] = len(dev_ids)
    settings["test_examples"] = len(test_ids)
    evaluation_folds = [
        evaluation_sample(val_ids, args.final_eval_examples, 42, 1000 + fold)
        for fold, (_, val_ids) in enumerate(
            validation_folds(
                examples, corpus.sentences, dev_ids, folds=2, seed=42, val_fraction=0.2
            ),
            1,
        )
    ]
    test_eval_ids = evaluation_sample(test_ids, args.final_eval_examples, 42, 2000)
    settings["validation_evaluation_examples"] = [len(ids) for ids in evaluation_folds]
    settings["test_evaluation_examples"] = len(test_eval_ids)
    setups = SETUPS if args.profile == "ablation" else HYPERPARAMETER_SETUPS
    write_json(
        args.output / "manifest.json",
        {"settings": settings, "setups": setups, "seeds": SEEDS, "corpus": corpus.summary},
    )
    results = []
    for setup in setups:
        setup_root = args.output / setup["name"]
        fold_rows = []
        splits = list(
            validation_folds(
                examples, corpus.sentences, dev_ids, folds=2, seed=42, val_fraction=0.2
            )
        )
        for fold, ((train_ids, val_ids), evaluation_ids) in enumerate(
            zip(splits, evaluation_folds, strict=True), 1
        ):
            for seed in (SEEDS[fold - 1],):
                torch.manual_seed(seed)
                model = QuantumAttentionModel(
                    len(corpus.vocabulary),
                    window=4,
                    embedding_dim=16,
                    qubits=4,
                    layers=setup["layers"],
                    alpha=setup.get("alpha", 0.05),
                    encoding=setup["encoding"],
                    circuit=setup["circuit"],
                ).to(args.device)
                history, full_validation = fit(
                    model,
                    contexts,
                    targets,
                    train_ids,
                    val_ids,
                    evaluation_ids,
                    settings | {"learning_rate": setup.get("learning_rate", args.learning_rate)},
                    seed,
                    setup_root / f"fold-{fold:02d}",
                )
                fold_rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "train_examples": len(train_ids),
                        "validation_examples": len(val_ids),
                        "evaluation_examples": len(evaluation_ids),
                        "full_validation": full_validation,
                        "best_monitor": best_validation_row(history)["validation"],
                    }
                )
        torch.manual_seed(42)
        model = QuantumAttentionModel(
            len(corpus.vocabulary),
            window=4,
            embedding_dim=16,
            qubits=4,
            layers=setup["layers"],
            alpha=setup.get("alpha", 0.05),
            encoding=setup["encoding"],
            circuit=setup["circuit"],
        ).to(args.device)
        history, _ = fit(
            model,
            contexts,
            targets,
            dev_ids,
            dev_ids,
            evaluation_sample(dev_ids, args.final_eval_examples, 42, 3000),
            settings | {"learning_rate": setup.get("learning_rate", args.learning_rate)},
            42,
            setup_root / "refit",
        )
        test = evaluate(model, contexts, targets, test_eval_ids, args.batch_size)
        test["source_examples"] = len(test_ids)
        test["evaluation_sampling"] = (
            "fixed seeded sample without replacement"
            if len(test_eval_ids) < len(test_ids)
            else "complete held-out split"
        )
        validation_ce = np.array([row["full_validation"]["cross_entropy"] for row in fold_rows])
        validation_top1 = np.array([row["full_validation"]["cosine_top1"] for row in fold_rows])
        validation_top5 = np.array([row["full_validation"]["cosine_top5"] for row in fold_rows])
        results.append(
            {
                "setup": setup,
                "folds": fold_rows,
                "validation": {
                    "cross_entropy_mean": float(validation_ce.mean()),
                    "cross_entropy_std": float(validation_ce.std(ddof=1)),
                    "cosine_top1_mean": float(validation_top1.mean()),
                    "cosine_top1_std": float(validation_top1.std(ddof=1)),
                    "cosine_top5_mean": float(validation_top5.mean()),
                    "cosine_top5_std": float(validation_top5.std(ddof=1)),
                },
                "test": test,
                "validation_evaluation_examples": [len(ids) for ids in evaluation_folds],
                "test_evaluation_examples": len(test_eval_ids),
                "final_evaluation": (
                    "fixed seeded samples without replacement"
                    if args.final_eval_examples
                    else "complete validation/test splits"
                ),
                "refit_best_monitor": best_validation_row(history)["validation"],
            }
        )
        write_json(setup_root / "summary.json", results[-1])
    write_json(args.output / "results.json", {"settings": settings, "results": results})
    print(f"Saved cross-validation results under {args.output}", flush=True)


if __name__ == "__main__":
    main()
