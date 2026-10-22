"""Validation-only fixed-walk/time and smoothing controls, using sparse evolution."""

from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from .data import graph_matrix, write_json
from .models import context_distribution
from .run import metrics


def run():
    torch.set_num_threads(1)
    root = Path(__file__).parent / "results"
    data = dict(np.load(root / "tatoeba_data" / "dataset.npz"))
    h, _ = graph_matrix(data["cooc"])
    c = context_distribution(torch.tensor(data["validation_contexts"]), len(h)).numpy()
    initial = np.sqrt(c).astype(complex)
    records = []
    for time in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        state = expm_multiply(-1j * time * csr_matrix(h), initial.T, traceA=0).T
        born = abs(state) ** 2
        born /= born.sum(1, keepdims=True)
        for mixture in [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            p = (1 - mixture) * born + mixture * data["unigram"]
            score, _ = metrics(p, data["validation_targets"])
            records.append({"time": time, "mixture": mixture, "validation": score})
    result = {
        "selected": min(records, key=lambda r: r["validation"]["nll"]),
        "grid": records,
        "scope": "Validation-only; no test evaluation or inclusion in primary ranking.",
    }
    write_json(root / "static_walk.json", result)
    return result["selected"]


if __name__ == "__main__":
    print(run())
