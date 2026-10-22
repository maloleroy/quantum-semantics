"""Legacy focused selection must not corrupt the fixed-budget trainer."""

import numpy as np
import pytest

from qcse.data import make_examples
from qcse.model import QCSEModel
from qcse.training import TrainConfig, train, validation_selection_key, validation_stopping_key


def test_loss_tie_breaker_does_not_reset_retrieval_patience():
    a = dict(
        epoch=1, validation=dict(exact_word_accuracy=0.1, paper_similarity_accuracy=0.5, bce=1.0)
    )
    b = dict(
        epoch=2, validation=dict(exact_word_accuracy=0.1, paper_similarity_accuracy=0.5, bce=0.9)
    )
    assert validation_selection_key(b) > validation_selection_key(a)
    assert validation_stopping_key(b) == validation_stopping_key(a)


def test_selected_weights_and_full_embeddings_agree(monkeypatch):
    import qcse.training as training

    vocabulary = list("abcde")
    examples = make_examples([list("abcde"), list("edcba")], vocabulary)
    model = QCSEModel(vocabulary, layers=1)
    initial = model.weights.copy()
    original = training.metrics
    calls = 0

    def scored(probabilities, targets):
        nonlocal calls
        result = original(probabilities, targets)
        result.update(exact_word_accuracy=0.5, paper_similarity_accuracy=0.8, bce=1.0)
        calls += 1
        return result

    monkeypatch.setattr(training, "metrics", scored)
    history, embeddings = train(
        model,
        examples,
        np.array([0, 1, 2, 3]),
        np.array([6, 7]),
        TrainConfig(epochs=4, batch_size=2),
        validation_ids=np.array([4, 5]),
        early_stopping_patience=1,
    )
    assert history[-1]["epoch"] == 1 and calls == 4
    np.testing.assert_array_equal(model.weights, initial)
    np.testing.assert_allclose(
        embeddings, model.predict_encoded([model.encode(e.context) for e in examples])
    )


def test_early_stopping_continuation_is_rejected_explicitly():
    with pytest.raises(ValueError, match="continuation"):
        train(None, [], [], [], initial_state={}, validation_ids=[], early_stopping_patience=1)
