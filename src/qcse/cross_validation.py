"""Sentence-grouped cross-validation inside a held-out development/test split."""

import json
from dataclasses import replace

import numpy as np

from .data import validation_folds, word_bits
from .outputs import save_npz, write_json
from .state_cache import ContextStates
from .training import load_run, metrics, save_run, train


def _fit(
    model,
    examples,
    train_ids,
    validation_ids,
    test_ids,
    config,
    output,
    original_ids,
    provenance,
    state_cache_mib,
):
    """Resume the last complete epoch, or return an already completed fit."""
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "run.npz"
    initial = None
    if checkpoint_path.exists():
        saved = load_run(checkpoint_path)
        model.weights = saved["weights"]
        initial = saved["state"]
    completed = initial["epoch"] if initial else 0
    if completed > config.epochs:
        raise ValueError("Checkpoint exceeds the configured cross-validation epoch target")

    def checkpoint(state):
        save_run(
            checkpoint_path,
            model,
            state,
            config=config,
            examples=examples,
            train_ids=train_ids,
            validation_ids=validation_ids,
            test_ids=test_ids,
            original_ids=original_ids,
            provenance=provenance,
            evaluate_test=False,
        )
        write_json(output / "history.json", state["history"])

    def progress(row):
        validation = row.get("validation")
        detail = f", validation BCE={validation['bce']:.8f}" if validation else ""
        scope = " (sample)" if "metric_examples" in row else ""
        print(
            f"{output.name} epoch {row['epoch']:3d}{scope}: "
            f"train BCE={row['train']['bce']:.8f}{detail}",
            flush=True,
        )

    if completed < config.epochs:
        history, embeddings = train(
            model,
            examples,
            train_ids,
            test_ids,
            replace(config, epochs=config.epochs - completed),
            callback=progress,
            initial_state=initial,
            checkpoint_callback=checkpoint,
            validation_ids=validation_ids,
            evaluate_test=False,
            state_cache_mib=state_cache_mib,
        )
    else:
        assert initial is not None
        history, embeddings = initial["history"], initial["embeddings"]
    model.save(output / "model.npz")
    write_json(output / "history.json", history)
    return history, embeddings


def cross_validate(
    model,
    examples,
    sentences,
    development_ids,
    test_ids,
    config,
    output,
    original_ids,
    provenance,
    folds=5,
    state_cache_mib=256,
    val_fraction=None,
):
    """Five independent fits, then a fresh fit on all development data.

    Fold histories contain train/validation only. The final fit has train-only
    history; the outer test score is computed once, after all fits are complete.
    Every fit has a portable checkpoint. Calling again in the same output root
    resumes interrupted fits and skips completed ones.
    """
    result_path = output / "cross_validation.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["status"] == "complete":
            print(f"Cross-validation already complete: {output}", flush=True)
            return result
    split_folds = list(
        validation_folds(examples, sentences, development_ids, folds, config.seed, val_fraction)
    )
    split_arrays = {
        "development_ids": development_ids,
        "test_ids": test_ids,
        "original_example_ids": original_ids,
    }
    for i, (train_ids, val_ids) in enumerate(split_folds, 1):
        split_arrays[f"fold_{i}_train_ids"] = train_ids
        split_arrays[f"fold_{i}_validation_ids"] = val_ids
    save_npz(output / "splits.npz", **split_arrays)
    development_examples = [examples[i] for i in development_ids]
    # Fold checkpoints address their development-only example arrays.
    local_ids = np.full(len(examples), -1, dtype=np.int64)
    local_ids[development_ids] = np.arange(len(development_ids))
    initial_weights = model.weights.copy()
    fold_rows, histories = [], []
    for i, (train_ids, val_ids) in enumerate(split_folds, 1):
        model.weights = initial_weights.copy()
        history, _ = _fit(
            model,
            development_examples,
            local_ids[train_ids],
            local_ids[val_ids],
            np.array([], dtype=np.int64),
            config,
            output / f"fold-{i:02d}",
            original_ids[development_ids],
            provenance | {"fold": i, "folds": folds},
            state_cache_mib,
        )
        histories.append(history)
        full_validation = history[-1]["validation"]
        if config.samples_per_epoch is not None:
            path = output / f"fold-{i:02d}" / "full_validation.json"
            if path.exists():
                full_validation = json.loads(path.read_text())
            else:
                print(f"fold-{i:02d}: scoring all {len(val_ids)} validation examples", flush=True)
                predictions = ContextStates(
                    model, [examples[j] for j in val_ids], state_cache_mib
                ).predict()
                full_validation = metrics(
                    predictions, word_bits([examples[j].target for j in val_ids], model.qubits)
                )
                write_json(path, full_validation)
        fold_rows.append(
            {
                "fold": i,
                "train_examples": len(train_ids),
                "validation_examples": len(val_ids),
                "metrics": full_validation,
            }
        )
        write_json(output / "cross_validation.json", {"status": "running", "folds": fold_rows})
    aggregate = []
    for epoch in range(config.epochs + 1):
        row: dict = {"epoch": epoch}
        for split in ("train", "validation"):
            values = {
                key: np.array([h[epoch][split][key] for h in histories])
                for key in histories[0][epoch][split]
            }
            row[split] = {
                key: {"mean": float(value.mean()), "std": float(value.std(ddof=1))}
                for key, value in values.items()
            }
        aggregate.append(row)
    write_json(output / "cv_history.json", aggregate)
    model.weights = initial_weights.copy()
    _, embeddings = _fit(
        model,
        examples,
        development_ids,
        None,
        test_ids,
        config,
        output / "refit",
        original_ids,
        provenance | {"phase": "refit"},
        state_cache_mib,
    )
    print(f"refit: scoring all {len(test_ids)} held-out test examples", flush=True)
    test_predictions = (
        ContextStates(model, [examples[i] for i in test_ids], state_cache_mib).predict()
        if config.samples_per_epoch is not None
        else embeddings[test_ids]
    )
    test_metrics = metrics(
        test_predictions, word_bits([examples[i].target for i in test_ids], model.qubits)
    )
    validation_summary = {}
    for key in fold_rows[0]["metrics"]:
        values = np.array([row["metrics"][key] for row in fold_rows])
        validation_summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    result = {
        "status": "complete",
        "folds": fold_rows,
        "epochs_per_fit": config.epochs,
        "samples_per_epoch": config.samples_per_epoch,
        "epoch_metrics": "fixed samples" if config.samples_per_epoch is not None else "full splits",
        "development_examples": len(development_ids),
        "test_examples": len(test_ids),
        "validation": validation_summary,
        "test": test_metrics,
        "final_model": "refit/model.npz",
        "test_evaluation": "once after final refit",
    }
    save_npz(
        output / "test_embeddings.npz",
        probabilities=test_predictions,
        target_ids=[examples[i].target for i in test_ids],
        example_ids=test_ids,
        original_example_ids=original_ids[test_ids],
    )
    write_json(output / "cross_validation.json", result)
    print(f"Final held-out test BCE={test_metrics['bce']:.8f}; results: {output}", flush=True)
    return result
