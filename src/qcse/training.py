"""Mini-batch Adam with reproducible SPSA gradient estimates.

SPSA uses two objective evaluations per batch regardless of parameter count.
This is an explicit implementation choice: the paper specifies the loss and
learning rate but does not give a complete optimizer/gradient recipe.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Example, word_bits

RUN_FORMAT_VERSION = 1


def save_run(path, model, state, *, config, examples, train_ids, test_ids, original_ids):
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    np.savez_compressed(temporary, **payload)
    temporary.with_name(temporary.name + ".npz").replace(path)


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

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
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
):
    """Train for ``config.epochs`` additional epochs.

    ``initial_state`` is the state returned to ``checkpoint_callback``.  The
    callback runs after the baseline and every completed epoch, so a caller can
    persist the last complete epoch without touching the optimizer internals.
    """
    config = config or TrainConfig()
    if not len(train_ids) or not len(test_ids):
        raise ValueError("Both training and test sets must be nonempty")
    targets = word_bits([e.target for e in examples], model.qubits)
    # Fixed context encoding needs no optimization: cache it, including repeats.
    cache = {}
    states = []
    state_ids = []
    for e in examples:
        if e.context not in cache:
            cache[e.context] = len(states)
            states.append(model.encode(e.context))
        state_ids.append(cache[e.context])
    state_ids = np.asarray(state_ids)
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
        # Evaluate each unique context once; reuse the result for all epoch outputs.
        embeddings = model.predict_encoded(states)[state_ids]
        row = {
            "epoch": epoch,
            "train": metrics(embeddings[train_ids], targets[train_ids]),
            "test": metrics(embeddings[test_ids], targets[test_ids]),
        }
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

    if start_epoch == 0 and not history:
        record(0)
    for epoch in range(start_epoch + 1, start_epoch + config.epochs + 1):
        order = rng.permutation(train_ids)
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            batch_states = [states[i] for i in state_ids[batch]]
            step += 1
            delta = rng.choice([-1.0, 1.0], size=len(model.weights))
            c = config.perturbation / step**0.101
            plus = model.predict_encoded(batch_states, model.weights + c * delta)
            minus = model.predict_encoded(batch_states, model.weights - c * delta)
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
