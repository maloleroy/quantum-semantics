"""Encoding candidates; learned corpus statistics use training examples only."""

import numpy as np

from qcse.context import ContextConfig, context_matrix, encoding_angles

METHODS = (
    "exponential",
    "diagonal",
    "phase",
    "hash",
    "angular",
    "exponential_rzrx",
    "exponential_ry",
    "logrank_ry",
    "permuted_ry",
    "scaled_ry",
    "slots_id",
    "slots_fourier",
    "random_mean",
    "random_recency",
    "random_slots",
    "binary_recency",
    "ppmi_mean",
    "ppmi_recency",
    "ppmi_slots",
    "ppmi_affine",
    "learned_token",
    "amplitude_raw",
    "amplitude_ppmi",
    "amplitude_complex",
    "fourier_reupload",
    "bigram_angles",
    "constant_zero",
)


class Features:
    def __init__(self, examples, train_ids, vocabulary_size, qubits, seed=2718):
        self.examples = examples
        self.size = vocabulary_size
        self.qubits = qubits
        self.width = 2 * qubits
        self.window = max(len(e.context) for e in examples)
        rng = np.random.default_rng(seed)
        self.random = rng.uniform(-1, 1, (vocabulary_size, self.width))
        self.permutation = rng.permutation(vocabulary_size)
        self.target_bits = ((np.arange(vocabulary_size)[:, None] >> np.arange(qubits)) & 1).astype(
            float
        )
        counts = np.bincount([examples[i].target for i in train_ids], minlength=vocabulary_size)
        self.unigram = (counts + 0.5) / (counts.sum() + 0.5 * vocabulary_size)
        self.bit_prior = (counts @ self.target_bits + 0.5) / (counts.sum() + 1)
        transitions = np.zeros((vocabulary_size, vocabulary_size))
        cooc = np.zeros_like(transitions)
        for i in train_ids:
            e = examples[i]
            transitions[e.context[-1], e.target] += 1
            for age, token in enumerate(reversed(e.context)):
                cooc[token, e.target] += 1 / (age + 1)
        self.bigram = (transitions + 10 * self.unigram) / (
            transitions.sum(axis=1, keepdims=True) + 10
        )
        self.bigram_bits = (transitions @ self.target_bits + 10 * self.bit_prior) / (
            transitions.sum(axis=1, keepdims=True) + 10
        )
        # Directional PPMI of previous word versus future target; no validation/test counts.
        denominator = cooc.sum(axis=1, keepdims=True) * cooc.sum(axis=0, keepdims=True)
        ppmi = np.maximum(0, np.log((cooc * cooc.sum() + 1e-12) / (denominator + 1e-12)))
        # Deterministic randomized low-rank SVD using only NumPy; avoids full V^3 SVD.
        projection = rng.normal(size=(vocabulary_size, min(self.width + 12, vocabulary_size)))
        sketch = ppmi @ projection
        for _ in range(2):
            sketch, _ = np.linalg.qr(sketch)
            sketch = ppmi @ (ppmi.T @ sketch)
        q, _ = np.linalg.qr(sketch)
        u, singular, _ = np.linalg.svd(q.T @ ppmi, full_matrices=False)
        embedding = (q @ u[:, : self.width]) * np.sqrt(singular[: self.width])
        self.ppmi = np.pad(embedding, ((0, 0), (0, self.width - embedding.shape[1])))
        # Scale fitted only on context-token occurrences in the training partition.
        tokens = [t for i in train_ids for t in examples[i].context]
        self.ppmi = np.tanh(self.ppmi / (np.std(self.ppmi[tokens], axis=0) + 1e-6))

    def pool(self, table, mode):
        values = []
        for e in self.examples:
            t = table[list(e.context)]
            if mode == "slots":
                width = self.width // self.window
                x = np.zeros(self.width)
                for age, row in enumerate(reversed(t)):
                    x[age * width : (age + 1) * width] = row[:width]
            else:
                weights = np.ones(len(t)) if mode == "mean" else 0.5 ** np.arange(len(t))[::-1]
                x = weights @ t / weights.sum()
            values.append(x)
        return np.asarray(values)

    def build(self, method):
        count = len(self.examples)
        kind, entangle, dynamic, reupload = "ry", True, None, False
        if method in METHODS[:5] or method in (
            "exponential_rzrx",
            "exponential_ry",
            "logrank_ry",
            "permuted_ry",
            "scaled_ry",
        ):
            base = method if method in METHODS[:5] else "exponential"
            values = []
            for e in self.examples:
                ids = list(e.context)
                if method == "permuted_ry":
                    ids = self.permutation[ids]
                if method == "logrank_ry":
                    theta = 2 * np.pi * np.log1p(ids) / np.log(self.size)
                    pos = np.arange(len(ids))
                    decay = np.exp(-np.abs(pos[:, None] - pos[None, :]))
                    matrix = decay * np.sin(theta[:, None]) * np.cos(theta[None, :])
                    matrix += theta[:, None]
                else:
                    matrix = context_matrix(ids, self.size, ContextConfig(base))
                if method == "scaled_ry":
                    matrix = 0.25 * matrix
                values.append(encoding_angles(matrix, self.qubits))
            # This study fixes window=4 and V=729, hence exactly one layer for all lengths.
            angles = np.asarray(values)
            kind = (
                "paper"
                if method in METHODS[:5]
                else ("rzrx" if method == "exponential_rzrx" else "ry")
            )
        elif method in ("slots_id", "slots_fourier", "fourier_reupload"):
            x = np.zeros((count, self.width))
            slot_width = self.width // self.window
            for row, e in enumerate(self.examples):
                for age, token in enumerate(reversed(e.context)):
                    theta = 2 * np.pi * token / self.size
                    start = age * slot_width
                    if method == "slots_id":
                        # Explicit presence channel distinguishes ID 0 from absent padding.
                        # Each scalar uses RY, not an RZ acting on an untouched |0>.
                        x[row, 2 * age] = theta
                        x[row, 2 * (self.window + age)] = 1
                    else:
                        x[row, start : start + slot_width] = [
                            np.sin(theta),
                            np.cos(theta),
                            np.sin(17 * theta),
                            np.cos(17 * theta),
                            1,
                        ][:slot_width]
            angles = x.reshape(count, 1, self.qubits, 2)
            reupload = method == "fourier_reupload"
        elif method == "binary_recency":
            pooled = self.pool(self.target_bits, "recency")
            last = self.target_bits[[e.context[-1] for e in self.examples]]
            angles = np.stack((np.pi / 2 + 0.8 * (2 * pooled - 1), 0.8 * (2 * last - 1)), -1)
            angles = angles[:, None]
        elif method == "constant_zero":
            angles = np.zeros((count, 1, self.qubits, 2))
            entangle = False
        elif method == "bigram_angles":
            p = self.bigram_bits[[e.context[-1] for e in self.examples]]
            angles = np.stack((2 * np.arcsin(np.sqrt(p)), np.zeros_like(p)), -1)[:, None]
            entangle = False
        elif method.startswith("amplitude_"):
            state = np.zeros((count, 1 << self.qubits), dtype=complex)
            if method == "amplitude_complex":
                for row, e in enumerate(self.examples):
                    for age, token in enumerate(reversed(e.context)):
                        state[row, token] += 0.7**age * np.exp(1j * age * np.pi / 2)
            elif method == "amplitude_raw":
                state[:, 0] = 1  # Anchor preserves scale and handles the all-zero context.
                for row, e in enumerate(self.examples):
                    x = context_matrix(e.context, self.size).ravel()
                    state[row, 1 : 1 + len(x)] = x
            else:
                state[:, 0] = 1
                state[:, 1 : 1 + self.width] = self.pool(self.ppmi, "recency")
            state /= np.linalg.norm(state, axis=1, keepdims=True)
            return {
                "state": state,
                "kind": "amplitude",
                "entangle": False,
                "angles": None,
                "dynamic": None,
                "reupload": False,
            }
        else:
            family = "ppmi" if method.startswith("ppmi") else "random"
            mode = method.rsplit("_", 1)[-1]
            mode = mode if mode in ("mean", "slots") else "recency"
            table = self.ppmi if family == "ppmi" else self.random
            angles = self.pool(table, mode).reshape(count, 1, self.qubits, 2)
            if method == "ppmi_affine":
                dynamic = "affine"
            elif method == "learned_token":
                dynamic = "table"
        return {
            "state": None,
            "angles": angles,
            "kind": kind,
            "entangle": entangle,
            "dynamic": dynamic,
            "reupload": reupload,
        }
