"""Language encoding, direct Born prediction and stable exponential gradients."""

import json

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from experiments.hamiltonian_study.born_evidence import hermitian
from experiments.hamiltonian_study.language_born import (
    ClassicalWords,
    HamiltonianWords,
    HermitianEvolution,
    Job,
    Partition,
    encode,
    evaluate,
    fit,
    score,
)
from experiments.hamiltonian_study.run import metrics


@pytest.mark.parametrize("scale", [0.0, 1e-9, 0.3])
def test_spectral_exponential_and_adjoint_match_independent_references(scale):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    coefficients = (torch.randn(16, dtype=torch.float64) * scale).requires_grad_(True)
    matrix = hermitian(coefficients, 4)
    actual = HermitianEvolution.apply(matrix)
    expected = torch.matrix_exp(-1j * matrix)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(
        actual.detach().numpy(), expm(-1j * matrix.detach().numpy()), atol=1e-13
    )
    cotangent = torch.randn((4, 4), dtype=torch.complex128)
    actual_gradient = torch.autograd.grad(
        (actual.conj() * cotangent).real.sum(), coefficients, retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad((expected.conj() * cotangent).real.sum(), coefficients)[
        0
    ]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-12, rtol=1e-10)
    assert torch.autograd.gradcheck(
        lambda c: HermitianEvolution.apply(hermitian(c, 4)), (coefficients,)
    )


def test_encoding_preserves_order_and_handles_padding_repeats_and_word_permutations():
    contexts = torch.tensor([[-1, -1, 1, 2], [1, 2, 1, 2], [-1, -1, -1, 3]])
    states = encode(contexts, 8)
    torch.testing.assert_close(states.norm(dim=1), torch.ones(3, dtype=torch.float64))
    # Repeated words add as complex amplitudes before normalization, not as probabilities.
    expected = torch.zeros(8, dtype=torch.complex128)
    expected[1] = np.sqrt(0.7) * 1j + np.sqrt(0.7**3) * -1j
    expected[2] = 1 - 0.7
    torch.testing.assert_close(states[1], expected / expected.norm())
    torch.testing.assert_close(
        states[2].abs().square(), torch.nn.functional.one_hot(torch.tensor(3), 8).double()
    )
    assert not torch.allclose(states[0], encode(torch.tensor([[-1, -1, 2, 1]]), 8)[0])
    permutation = torch.tensor([4, 2, 6, 0, 1, 7, 3, 5])
    renamed = torch.where(contexts >= 0, permutation[contexts.clamp_min(0)], -1)
    torch.testing.assert_close(encode(renamed, 8)[:, permutation], states)
    with pytest.raises(ValueError, match="Empty"):
        encode(torch.full((1, 4), -1), 8)


def test_born_word_probabilities_match_density_evolution_and_have_no_learned_head():
    torch.manual_seed(23)
    states = encode(torch.tensor([[1, 2, 3, 1], [-1, -1, 2, 4]]), 8)
    coherent, dephased = HamiltonianWords(8), HamiltonianWords(8, True)
    dephased.load_state_dict(coherent.state_dict())
    unitary = coherent.operator().detach()
    for model in [coherent, dephased]:
        expected = []
        for state in states:
            density = (
                torch.diag(state.abs().square()).to(torch.complex128)
                if model.dephased
                else torch.outer(state, state.conj())
            )
            expected.append(torch.diag(unitary @ density @ unitary.mH).real)
        actual = model(states, torch.tensor([1, 4]))
        torch.testing.assert_close(actual, torch.stack(expected), atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(actual.sum(1), torch.ones(2, dtype=torch.float64))
        assert [name for name, _ in model.named_parameters()] == ["coefficients"]
        assert model.coefficients.numel() == 64
    phase_changed = states * torch.exp(1j * torch.arange(8, dtype=torch.float64))
    torch.testing.assert_close(dephased(states, None), dephased(phase_changed, None))
    assert not torch.allclose(coherent(states, None), coherent(phase_changed, None))


def test_streamed_scoring_matches_full_metrics_and_fitting_never_requires_test_arrays(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(31)
    contexts = rng.integers(0, 8, (40, 4))
    targets = rng.integers(0, 8, 40)
    part = Partition(
        encode(torch.tensor(contexts), 8),
        torch.tensor(targets),
        torch.tensor(contexts[:, -1]),
        np.arange(40),
    )
    model = ClassicalWords(8)
    result, arrays = score(model, part)
    expected, expected_arrays = metrics(
        model(part.states, part.last_words).detach().numpy(), targets
    )
    for key in expected:
        assert result[key] == pytest.approx(expected[key])
    for key in arrays:
        np.testing.assert_allclose(arrays[key], expected_arrays[key])
    np.savez(
        tmp_path / "dataset.npz",
        train_contexts=contexts[:30],
        train_targets=targets[:30],
        train_groups=np.arange(30),
        validation_contexts=contexts[30:],
        validation_targets=targets[30:],
        validation_groups=np.arange(30, 40),
        vocabulary=np.array([f"w{i}" for i in range(8)]),
        unigram=np.full(8, 1 / 8),
        transitions=np.ones((8, 8)),
    )
    hashes = []
    for kind in ["coherent", "dephased"]:
        result = fit(Job(str(tmp_path), str(tmp_path / kind), kind, 17, 0.003, 3, 16))
        hashes.append(result["initial_parameter_hashes"])
        assert result["selected_validation"]["nll"] == min(
            row["validation"]["nll"] for row in result["history"]
        )
        assert result["first_gradient_norm"] > 0
        assert json.loads((tmp_path / kind / "result.json").read_text())["parameters"] == 64
    assert hashes[0] == hashes[1]
    with pytest.raises(FileNotFoundError, match="selection_lock"):
        evaluate(tmp_path)
