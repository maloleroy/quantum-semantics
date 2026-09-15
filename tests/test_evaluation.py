import numpy as np
import pytest

from qcse.evaluation import evaluation_sample


def test_evaluation_sample_is_deterministic_sorted_and_without_replacement():
    ids = np.arange(100)
    first = evaluation_sample(ids, 10, seed=42, stream=2000)
    second = evaluation_sample(ids, 10, seed=42, stream=2000)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 10
    assert len(np.unique(first)) == 10
    assert np.all(np.diff(first) > 0)
    assert set(first) <= set(ids)


def test_evaluation_sample_returns_all_when_uncapped_or_smaller():
    ids = np.array([7, 2, 9])
    np.testing.assert_array_equal(evaluation_sample(ids, 10, 42, 1), ids)
    np.testing.assert_array_equal(evaluation_sample(ids, 0, 42, 1), ids)


def test_evaluation_sample_rejects_negative_limit():
    with pytest.raises(ValueError, match="nonnegative"):
        evaluation_sample(np.arange(4), -1, 42, 1)
