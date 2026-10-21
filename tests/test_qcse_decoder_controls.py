"""Verify the fixed-code log-linear interpretation against the actual adapter."""

import pytest
import torch
from torch.nn import functional as F

from qcse.benchmark_models import PaperQCSE


@pytest.mark.parametrize("vocabulary", [5, 8])
def test_word_softmax_and_gradients_equal_fixed_binary_code_head(vocabulary):
    # V=5 also checks renormalization with three invalid q3 bitstrings removed.
    model = PaperQCSE(vocabulary, dict(layers=1)).double()
    logits = torch.linspace(-1.7, 2.1, 12, dtype=torch.float64).reshape(4, 3)
    logits.requires_grad_()
    model.probabilities = lambda contexts: logits.sigmoid()
    actual = model(torch.zeros(4, 1, dtype=torch.long))
    fixed_code = logits @ model.word_bits.T
    torch.testing.assert_close(
        actual.log_softmax(-1), fixed_code.log_softmax(-1), atol=1e-14, rtol=1e-14
    )
    targets = torch.tensor([0, 1, 2, vocabulary - 1])
    first = torch.autograd.grad(F.cross_entropy(actual, targets), logits, retain_graph=True)[0]
    second = torch.autograd.grad(F.cross_entropy(fixed_code, targets), logits)[0]
    torch.testing.assert_close(first, second, atol=1e-14, rtol=1e-14)


def test_joint_born_probabilities_are_not_identified_by_marginals():
    # Bell distribution and uniform independent bits share marginal p=(.5,.5).
    bell = torch.tensor([0.5, 0.0, 0.0, 0.5])
    independent = torch.full((4,), 0.25)
    bits = ((torch.arange(4)[:, None] >> torch.arange(2)) & 1).float()
    torch.testing.assert_close(bell @ bits, independent @ bits)
    assert not torch.equal(bell, independent)
