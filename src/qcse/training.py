"""Mini-batch Adam with reproducible SPSA gradient estimates.

SPSA uses two objective evaluations per batch regardless of parameter count.
This is an explicit implementation choice: the paper specifies the loss and
learning rate but does not give a complete optimizer/gradient recipe.
"""

import json
from dataclasses import dataclass

import numpy as np

from .data import Example, word_bits
from .outputs import save_npz
from .state_cache import ContextStates

RUN_FORMAT_VERSION = 1


def save_run(
    path,
    model,
    state,
    *,
    config,
    examples,
    train_ids,
    test_ids,
    original_ids,
    provenance=None,
    validation_ids=None,
    evaluate_test=True,
):
    """Save a complete, resumable training run in one compressed archive."""
    metadata = {
        "format_version": RUN_FORMAT_VERSION,
        "model": {
            "vocabulary": model.vocabulary,
            "layers": model.layers,
            "context": model.context.__dict__,
            "window": model.window,
            "objective": model.objective,
            "direction": model.direction,
        },
        "config": config.__dict__,
        "provenance": provenance or {},
        "evaluation": "validation"
        if validation_ids is not None
        else "test"
        if evaluate_test
        else "none",
        "embeddings_scope": "omitted" if config.samples_per_epoch is not None else "all examples",
        "step": int(state["step"]),
        "examples": [
            {
                "sentence": e.sentence,
                "position": e.position,
                "target": e.target,
                "context": list(e.context),
            }
            for e in examples
        ],
    }
    payload = {
        "metadata": np.array(json.dumps(metadata)),
        "history": np.array(json.dumps(state["history"])),
        "rng_state": np.array(json.dumps(state["rng_state"])),
        "weights": model.weights,
        "first": state["first"],
        "second": state["second"],
        "embeddings": state["embeddings"],
        "original_example_ids": np.asarray(original_ids),
        "train_ids": np.asarray(train_ids),
        "test_ids": np.asarray(test_ids),
    }
    if validation_ids is not None:
        payload["validation_ids"] = np.asarray(validation_ids)
    save_npz(path, **payload)


def load_run(path):
    """Load a run archive, returning metadata, arrays and JSON state."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.pop("format_version") != RUN_FORMAT_VERSION:
            raise ValueError("Unsupported training run format")
        state = {
            "epoch": int(json.loads(str(archive["history"]))[-1]["epoch"]),
            "step": int(metadata["step"]),
            "first": archive["first"].copy(),
            "second": archive["second"].copy(),
            "rng_state": json.loads(str(archive["rng_state"])),
            "history": json.loads(str(archive["history"])),
            "embeddings": archive["embeddings"].copy(),
        }
        return {
            "metadata": metadata,
            "state": state,
            "weights": archive["weights"].copy(),
            "original_ids": archive["original_example_ids"].copy(),
            "train_ids": archive["train_ids"].copy(),
            "test_ids": archive["test_ids"].copy(),
            "validation_ids": archive["validation_ids"].copy()
            if "validation_ids" in archive
            else None,
            "examples": [
                Example(item["sentence"], item["position"], item["target"], tuple(item["context"]))
                for item in metadata["examples"]
            ],
        }


def binary_cross_entropy(probabilities, targets):
    p = np.clip(probabilities, 1e-10, 1 - 1e-10)
    return float(-np.mean(targets * np.log(p) + (1 - targets) * np.log1p(-p)))


def metrics(probabilities, targets):
    matching = (probabilities >= 0.5) == targets.astype(bool)
    return {
        "bce": binary_cross_entropy(probabilities, targets),
        "bit_accuracy": float(matching.mean()),
        "exact_word_accuracy": float(matching.all(axis=1).mean()),
        "paper_similarity_accuracy": float((matching.mean(axis=1) >= 0.5).mean()),
    }


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.0003
    l2: float = 0.001
    perturbation: float = 0.1
    seed: int = 42
    samples_per_epoch: int | None = None
    eval_examples: int = 2048

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.samples_per_epoch is not None and self.samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
        if self.eval_examples < 1:
            raise ValueError("eval_examples must be positive")
        if not all(np.isfinite(x) and x > 0 for x in (self.learning_rate, self.perturbation)):
            raise ValueError("learning_rate and perturbation must be positive and finite")
        if not np.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("l2 must be nonnegative and finite")


def train(
    model,
    examples,
    train_ids,
    test_ids,
    config=None,
    callback=None,
    initial_state=None,
    checkpoint_callback=None,
    *,
    validation_ids=None,
    evaluate_test=True,
    state_cache_mib=256,
):
    """Train for ``config.epochs`` additional epochs.

    With ``samples_per_epoch``, draw that many examples with replacement from
    the full training split. Histories use fixed monitoring samples and returned
    embeddings are empty; callers perform final full evaluation separately.

    ``initial_state`` is the state returned to ``checkpoint_callback``.  The
    callback runs after the baseline and every completed epoch, so a caller can
    persist the last complete epoch without touching the optimizer internals.
    """
    config = config or TrainConfig()
    evaluation_ids = validation_ids if validation_ids is not None else test_ids
    evaluation_name = "validation" if validation_ids is not None else "test"
    evaluate = validation_ids is not None or evaluate_test
    if not len(train_ids) or (evaluate and not len(evaluation_ids)):
        raise ValueError("Both training and evaluation sets must be nonempty")
    targets = word_bits([e.target for e in examples], model.qubits)
    states = ContextStates(model, examples, state_cache_mib)
    sampled = config.samples_per_epoch is not None
    # Fixed monitoring samples use an independent RNG so reporting does not
    # consume training draws. Recreating them on resume gives the same IDs.
    monitor_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))

    def monitor_ids(ids):
        if sampled and len(ids) > config.eval_examples:
            return monitor_rng.choice(ids, config.eval_examples, replace=False)
        return ids

    train_monitor = monitor_ids(train_ids)
    evaluation_monitor = monitor_ids(evaluation_ids) if evaluate else np.array([], dtype=np.int64)
    if initial_state is None:
        rng = np.random.default_rng(config.seed)
        first = np.zeros_like(model.weights)
        second = np.zeros_like(model.weights)
        history = []
        start_epoch = 0
        step = 0
    else:
        rng = np.random.default_rng()
        rng.bit_generator.state = initial_state["rng_state"]
        first = np.asarray(initial_state["first"], dtype=float).copy()
        second = np.asarray(initial_state["second"], dtype=float).copy()
        history = list(initial_state["history"])
        start_epoch = int(initial_state["epoch"])
        step = int(initial_state["step"])
        if first.shape != model.weights.shape or second.shape != model.weights.shape:
            raise ValueError("Saved optimizer state does not match model weights")
        if not history or history[-1]["epoch"] != start_epoch:
            raise ValueError("Saved training history does not match checkpoint epoch")

    def record(epoch):
        # Sampled epochs must not trigger a hidden full-corpus prediction pass.
        embeddings = np.empty((0, model.qubits)) if sampled else states.predict()
        train_probabilities = states.predict(train_monitor) if sampled else embeddings[train_ids]
        row = {
            "epoch": epoch,
            "train": metrics(train_probabilities, targets[train_monitor]),
        }
        if evaluate:
            probabilities = (
                states.predict(evaluation_monitor) if sampled else embeddings[evaluation_ids]
            )
            row[evaluation_name] = metrics(probabilities, targets[evaluation_monitor])
        if sampled:
            row["metric_examples"] = {"train": len(train_monitor)}
            if evaluate:
                row["metric_examples"][evaluation_name] = len(evaluation_monitor)
        history.append(row)
        if checkpoint_callback:
            checkpoint_callback(
                {
                    "epoch": epoch,
                    "step": step,
                    "first": first.copy(),
                    "second": second.copy(),
                    "rng_state": rng.bit_generator.state,
                    "history": list(history),
                    "embeddings": embeddings,
                }
            )
        if callback:
            callback(row)
        return embeddings

    embeddings = initial_state["embeddings"] if initial_state else record(0)
    for epoch in range(start_epoch + 1, start_epoch + config.epochs + 1):
        if config.samples_per_epoch is None:
            order = rng.permutation(train_ids)
        else:
            order = rng.choice(train_ids, config.samples_per_epoch, replace=True)
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            step += 1
            delta = rng.choice([-1.0, 1.0], size=len(model.weights))
            c = config.perturbation / step**0.101
            plus, minus = states.predict(
                batch, np.stack((model.weights + c * delta, model.weights - c * delta))
            )
            gradient = (
                (
                    binary_cross_entropy(plus, targets[batch])
                    - binary_cross_entropy(minus, targets[batch])
                )
                / (2 * c)
            ) * delta
            # Loss convention: mean bitwise BCE + lambda * sum(theta**2).
            gradient += 2 * config.l2 * model.weights
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            model.weights -= (
                config.learning_rate
                * (first / (1 - 0.9**step))
                / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            )
        embeddings = record(epoch)
    return history, embeddings
