"""The five context encodings: Eq. (24) and Appendix Eqs. (25)-(28)."""

import math
from dataclasses import dataclass

import numpy as np

METHODS = ("exponential", "diagonal", "phase", "hash", "angular")


@dataclass(frozen=True)
class ContextConfig:
    method: str = "exponential"
    alpha: float = 1.0
    omega: float = 1.0
    delta: float = 1.0
    prime: int = 31
    hash_size: int = 997

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"Unknown context method: {self.method}")
        if not all(np.isfinite(x) and x > 0 for x in (self.alpha, self.omega, self.delta)):
            raise ValueError("alpha, omega and delta must be finite and positive")
        if (
            self.hash_size < 1
            or self.prime < 2
            or any(self.prime % d == 0 for d in range(2, math.isqrt(self.prime) + 1))
        ):
            raise ValueError("hash_size must be positive and prime must be prime")


def context_matrix(indices, vocabulary_size: int, config: ContextConfig | None = None):
    config = config or ContextConfig()
    ids = np.asarray(indices, dtype=int)
    if ids.ndim != 1 or not len(ids):
        raise ValueError("A nonempty one-dimensional context is required")
    if vocabulary_size < 1 or np.any(ids < 0) or np.any(ids >= vocabulary_size):
        raise ValueError("Context IDs must belong to the vocabulary")
    theta = ids * (2 * np.pi / vocabulary_size)
    positions = np.arange(len(ids))
    decay = np.exp(-config.alpha * np.abs(positions[:, None] - positions[None, :]))
    if config.method == "angular":
        # Appendix leaves square padding unspecified: pad to the next square.
        side = math.ceil(math.sqrt(len(ids)))
        return np.pad(config.omega * theta, (0, side * side - len(ids))).reshape(side, side)
    if config.method == "exponential":
        return (
            decay * np.sin(config.omega * theta[:, None]) * np.cos(config.omega * theta[None, :])
            + theta[:, None]
        )
    if config.method == "diagonal":
        matrix = decay * np.sin(config.omega * theta[:, None]) + theta[:, None]
        np.fill_diagonal(matrix, np.log1p(ids))
        return matrix
    modulation = (
        config.delta * theta
        if config.method == "phase"
        else (ids * config.prime) % config.hash_size
    )
    return decay * np.sin((config.omega * positions + modulation)[:, None]) + theta[:, None]


def encoding_angles(matrix, qubits: int) -> np.ndarray:
    """Row-major flatten, zero-pad, consume 2m angles per layer.

    Shape (L, m, 2): adjacent values feed RX then RZ of a qubit. The paper's
    C-tilde (2m, L) is angles.reshape(L, 2m).T, i.e. sequential chunks as columns.
    """
    if qubits < 1:
        raise ValueError("At least one qubit is required")
    flat = np.asarray(matrix, dtype=float).ravel()
    if not flat.size or not np.isfinite(flat).all():
        raise ValueError("Context matrix must be nonempty and finite")
    return np.pad(flat, (0, (-flat.size) % (2 * qubits))).reshape(-1, qubits, 2)
