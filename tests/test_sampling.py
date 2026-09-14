from dataclasses import replace
from math import ceil

import numpy as np
import pytest
import torch

from qcse.data import make_examples, split_examples
from qcse.model import QCSEModel
from qcse.state_cache import ContextStates
from qcse.training import TrainConfig, train

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")),
    pytest.param(
        "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="No MPS")
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("objective", ["causal", "cbow"])
@pytest.mark.parametrize("batch_size", [4, 6])
def test_sampled_epochs_bound_work_and_resume_exactly(monkeypatch, device, objective, batch_size):
    vocabulary = list("abcdefgh")
    sentences = [[a, b, c] for a in vocabulary for b in vocabulary for c in vocabulary]
    examples = make_examples(sentences, vocabulary, objective=objective)
    train_ids, test_ids = split_examples(examples, sentences)
    batches, monitors = [], []
    predict = ContextStates.predict

    def track(self, ids=None, weights=None):
        assert ids is not None, "Sampled epochs must never predict the full corpus"
        (batches if weights is not None else monitors).append(ids.copy())
        return predict(self, ids, weights)

    monkeypatch.setattr(ContextStates, "predict", track)
    model = QCSEModel(vocabulary, device=device, objective=objective)
    config = TrainConfig(epochs=3, batch_size=batch_size, samples_per_epoch=20, eval_examples=7)
    snapshots = []
    history, embeddings = train(
        model, examples, train_ids, test_ids, config, checkpoint_callback=snapshots.append
    )
    steps = ceil(20 / batch_size)
    assert len(batches) == 3 * steps
    for epoch in range(3):
        assert sum(len(ids) for ids in batches[epoch * steps : (epoch + 1) * steps]) == 20
    assert all(len(ids) <= batch_size and set(ids) <= set(train_ids) for ids in batches)
    assert (
        np.concatenate(batches).max() > 512
    )  # Sampling reaches beyond the former tiny prefix/cap.
    assert len(monitors) == 8  # Two fixed monitor subsets at epoch 0 and each of three epochs.
    for train_sample, test_sample in zip(monitors[::2], monitors[1::2], strict=True):
        assert len(train_sample) == len(test_sample) == 7
        np.testing.assert_array_equal(train_sample, monitors[0])
        np.testing.assert_array_equal(test_sample, monitors[1])
        assert not set(train_sample) & set(test_sample)
    assert not len(embeddings)
    assert snapshots[-1]["step"] == 3 * steps

    resumed = QCSEModel(vocabulary, device=device, objective=objective)
    first_epoch = []
    train(
        resumed,
        examples,
        train_ids,
        test_ids,
        replace(config, epochs=1),
        checkpoint_callback=first_epoch.append,
    )
    resumed_history, _ = train(
        resumed,
        examples,
        train_ids,
        test_ids,
        replace(config, epochs=2),
        initial_state=first_epoch[-1],
    )
    np.testing.assert_array_equal(resumed.weights, model.weights)
    assert resumed_history == history


@pytest.mark.parametrize(
    "kwargs", [{"samples_per_epoch": 0}, {"samples_per_epoch": -1}, {"eval_examples": 0}]
)
def test_invalid_sampling_budget(kwargs):
    with pytest.raises(ValueError, match="positive"):
        TrainConfig(**kwargs)
