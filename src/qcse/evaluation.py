"""Deterministic bounded evaluation subsets."""

import numpy as np


def evaluation_sample(ids, limit, seed, stream):
    """Return a reproducible subset without replacement, or all IDs when uncapped."""
    ids = np.asarray(ids, dtype=np.int64)
    if limit is None or limit == 0 or len(ids) <= limit:
        return ids.copy()
    if limit < 0:
        raise ValueError("evaluation sample size must be nonnegative")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))
    return np.sort(rng.choice(ids, size=limit, replace=False))
