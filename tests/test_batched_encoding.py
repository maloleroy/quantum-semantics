"""Parity with the scalar/Qiskit encoder and bounded cache misses."""

import numpy as np
import pytest
import torch

from qcse.context import (
    METHODS,
    ContextConfig,
    context_matrices,
    context_matrix,
    encoding_angles,
    encoding_angles_batch,
)
from qcse.data import make_examples
from qcse.model import QCSEModel
from qcse.state_cache import ContextStates


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("length", [1, 3, 5])
def test_batched_contexts_preserve_scalar_formulas(method, length):
    ids = np.random.default_rng(42).integers(0, 9, (3, length))
    config = ContextConfig(method=method, alpha=0.3, omega=0.7, delta=1.3)
    matrices = context_matrices(ids, 9, config)
    expected = np.stack([context_matrix(row, 9, config) for row in ids])
    np.testing.assert_allclose(matrices, expected, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(
        encoding_angles_batch(matrices, 4), np.stack([encoding_angles(m, 4) for m in matrices])
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_mixed_length_states_match_qiskit_and_encoding_batches_are_bounded(
    method, direction, monkeypatch
):
    model = QCSEModel(
        list("abcdefghi"),
        layers=3,
        direction=direction,
        context=ContextConfig(method=method),
        simulation_batch_size=2,
    )
    contexts = [(0,), (2, 3, 4), (1, 2), (8, 0, 7), (2, 2, 2), (0,)]
    expected = np.array([model.encode(c).data for c in contexts])
    calls, original = [], model.simulator.encode

    def bounded(angles):
        calls.append(len(angles))
        return original(angles)

    monkeypatch.setattr(model.simulator, "encode", bounded)
    actual = model.encode_many(contexts)
    np.testing.assert_allclose(
        actual[..., 0].numpy() + 1j * actual[..., 1].numpy(), expected, atol=1e-12, rtol=0
    )
    assert max(calls) <= 2 and sum(calls) == len(contexts)
    assert model.encode_many([]).shape == (0, 16, 2)


def test_cache_eviction_retains_order_duplicates_and_independent_host_storage(monkeypatch):
    vocabulary = list("abcde")
    model = QCSEModel(vocabulary, layers=2, simulation_batch_size=2)
    examples = make_examples([list("abcde"), list("ecdab")], vocabulary, window=2)
    cache = ContextStates(model, examples)
    cache.packed, cache.capacity = None, 1
    calls, original = [], model.encode_many

    def counted(contexts):
        calls.append(list(contexts))
        return original(contexts)

    monkeypatch.setattr(model, "encode_many", counted)
    for ids in ([0, 1, 0], [0, 2, 1]):
        actual = cache._states(ids)
        expected = model.prepare_states([model.encode(cache.contexts[i]) for i in ids])
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        assert len(cache.cache) == 1
        for state in cache.cache.values():
            assert state.device.type == "cpu"
            assert state.untyped_storage().nbytes() == state.numel() * state.element_size()
    assert len(calls[0]) == 2 and len(calls[1]) == 2


def test_empty_matrix_axes_are_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        encoding_angles_batch(np.empty((2, 3, 0)), 4)
