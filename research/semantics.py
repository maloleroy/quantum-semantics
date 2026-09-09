"""Simple semantic codes and train-only classical controls."""

import numpy as np

from qcse.data import word_bits


def semantic_codes(vectors, qubits):
    """Balanced recursive semantic partition; unique binary leaf codes.

    At each node, split by projection onto two far-apart vocabulary vectors.
    Root decisions occupy high bits. No attribute names or target labels are used.
    """
    if len(vectors) > 2**qubits:
        raise ValueError("Not enough bits for unique codes")
    codes = np.zeros(len(vectors), dtype=int)

    def partition(ids, bit):
        if len(ids) <= 1:
            return
        if bit < 0:
            raise ValueError("Code collision")
        x = vectors[ids]
        a = int(np.argmin(x @ x[0]))
        b = int(np.argmin(x @ x[a]))
        score = x @ (x[b] - x[a])
        order = ids[np.argsort(score, kind="stable")]
        half = len(order) // 2
        left, right = order[:half], order[half:]
        codes[right] |= 1 << bit
        partition(left, bit - 1)
        partition(right, bit - 1)

    partition(np.arange(len(vectors)), qubits - 1)
    assert len(np.unique(codes)) == len(vectors)
    return codes


def code_geometry(vectors, codes, qubits):
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argsort(-similarity, axis=1)[:, :5]
    bits = word_bits(codes, qubits)
    near = np.count_nonzero(bits[:, None, :] != bits[neighbors], axis=2)
    rng = np.random.default_rng(911)
    # Uniform different-word pairs, fixed for every codebook.
    i = rng.integers(len(codes), size=10000)
    j = (i + rng.integers(1, len(codes), size=10000)) % len(codes)
    random = np.count_nonzero(bits[i] != bits[j], axis=1)
    return {
        "nearest5_mean_hamming": float(near.mean()),
        "nearest5_within3_fraction": float((near <= 3).mean()),
        "random_pair_mean_hamming": float(random.mean()),
        "random_pair_within3_fraction": float((random <= 3).mean()),
        "unique_codes": len(np.unique(codes)),
    }


def fit_projection(vectors, examples, train_ids, dimensions):
    """PCA fitted on occurrence-weighted TRAIN context means, then fixed angle scaling."""
    means = np.array([vectors[list(e.context)].mean(axis=0) for e in examples])
    center = means[train_ids].mean(axis=0)
    _, _, axes = np.linalg.svd(means[train_ids] - center, full_matrices=False)
    axes = axes[:dimensions].T
    scores = (means[train_ids] - center) @ axes
    scale = np.maximum(scores.std(axis=0), 1e-6)
    # Linear projection commutes with context averaging. No clipping before averaging.
    projected = (vectors - center) @ axes / scale * 0.5
    return projected, {"center": center, "axes": axes, "scale": scale}


def fit_linear(features, targets, train_ids, steps=400, rate=0.05, l2=0.001):
    """Full-batch logistic regression, fixed budget, train-only normalization."""
    center = features[train_ids].mean(axis=0)
    scale = np.maximum(features[train_ids].std(axis=0), 1e-6)
    x = (features - center) / scale
    x = np.column_stack([x, np.ones(len(x))])
    weights = np.zeros((x.shape[1], targets.shape[1]))
    first = np.zeros_like(weights)
    second = np.zeros_like(weights)
    for step in range(1, steps + 1):
        logits = np.clip(x[train_ids] @ weights, -35, 35)
        p = 1 / (1 + np.exp(-logits))
        gradient = x[train_ids].T @ (p - targets[train_ids]) / targets[train_ids].size
        gradient[:-1] += 2 * l2 * weights[:-1]
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        weights -= rate * (first / (1 - 0.9**step)) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
    probabilities = 1 / (1 + np.exp(-np.clip(x @ weights, -35, 35)))
    return probabilities, {"weights": weights, "center": center, "scale": scale}
