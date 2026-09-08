"""Mini-batch Adam with reproducible SPSA gradient estimates.

SPSA uses two objective evaluations per batch regardless of parameter count.
This is an explicit implementation choice: the paper specifies the loss and
learning rate but does not give a complete optimizer/gradient recipe.
"""

from dataclasses import dataclass

import numpy as np

from .data import word_bits


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


def train(model, examples, train_ids, test_ids, config=None, callback=None):
    config = config or TrainConfig()
    if not len(train_ids) or not len(test_ids):
        raise ValueError("Both training and test sets must be nonempty")
    targets = word_bits([e.target for e in examples], model.qubits)
    # Fixed context encoding needs no optimization: cache it, including repeats.
    cache = {}
    states = []
    for e in examples:
        if e.context not in cache:
            cache[e.context] = model.encode(e.context)
        states.append(cache[e.context])
    rng = np.random.default_rng(config.seed)
    first = np.zeros_like(model.weights)
    second = np.zeros_like(model.weights)
    history = []
    step = 0

    def evaluate(ids):
        p = model.predict_encoded([states[i] for i in ids])
        return metrics(p, targets[ids])

    def record(epoch):
        row = {"epoch": epoch, "train": evaluate(train_ids), "test": evaluate(test_ids)}
        history.append(row)
        if callback:
            callback(row)

    record(0)
    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(train_ids)
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            batch_states = [states[i] for i in batch]
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
        record(epoch)
    return history, model.predict_encoded(states)
