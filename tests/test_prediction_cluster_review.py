import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/review_prediction_clusters.py"
    spec = importlib.util.spec_from_file_location("prediction_review", path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_cluster_multiplicities_match_raw_token_resampling_and_chunking():
    review = module()
    labels = np.array([2, 2, 5, 9, 9, 9])
    groups = np.unique(labels)
    raw = np.random.default_rng(8).normal(size=(2, 3, len(labels), 3))
    sums = np.stack([raw[:, :, labels == g].sum(2) for g in groups], axis=2)[None]
    counts = np.array([[(labels == g).sum() for g in groups]])
    paired, sentences = review.bootstrap(sums, counts, draws=100, seed=42, chunk=7)
    rng = np.random.default_rng(42)
    expected_pair, expected_sentence = [], []
    for _ in range(100):
        selected = rng.integers(len(groups), size=len(groups))
        seeds = rng.integers(3, size=3)
        ids = np.concatenate([np.flatnonzero(labels == groups[g]) for g in selected])
        expected_pair.append(raw[:, seeds][:, :, ids].mean((1, 2)))
        expected_sentence.append(raw[:, :, ids].mean((1, 2)))
    np.testing.assert_allclose(paired[:, 0], expected_pair, atol=1e-12, rtol=0)
    np.testing.assert_allclose(sentences[:, 0], expected_sentence, atol=1e-12, rtol=0)
    other, _ = review.bootstrap(sums, counts, draws=100, seed=42, chunk=13)
    np.testing.assert_allclose(other, paired, atol=1e-12, rtol=0)
    # Independent token bootstrap would give these unequal sentence sizes different weights.
    assert counts.tolist() == [[2, 1, 3]]


@pytest.mark.parametrize("bad_counts", [np.array([[0, 1]]), np.array([[np.nan, 1]])])
def test_invalid_group_weights_are_rejected(bad_counts):
    with pytest.raises(ValueError, match="finite sums and positive"):
        module().bootstrap(np.ones((1, 2, 3, 2, 3)), bad_counts, draws=10)
