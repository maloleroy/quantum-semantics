"""Independent checks for the fixed-encoding Hamiltonian experiment."""

import json

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from experiments.hamiltonian_study.analyze_born import dephased_nll_bound
from experiments.hamiltonian_study.born_evidence import (
    BornModel,
    Job,
    Spec,
    evaluate,
    fit,
    hermitian,
    metrics,
    phase_table,
    quantum_probabilities,
)


def test_born_evolution_and_gradients_match_independent_exponential():
    torch.set_num_threads(1)
    coefficients = torch.tensor(np.random.default_rng(19).normal(0, 0.2, 16), requires_grad=True)
    matrix = hermitian(coefficients, 4)
    torch.testing.assert_close(matrix, matrix.mH)
    unitary = torch.matrix_exp(-1j * matrix)
    inputs = torch.tensor([[0, 1], [2, 3]])
    probabilities = quantum_probabilities(inputs, unitary, Spec("modular", 4, 4))
    reference = expm(-1j * matrix.detach().numpy())
    expected = []
    for first, second in inputs.tolist():
        phase = np.diag(np.exp(2j * np.pi * np.arange(4) * second / 4))
        state = reference.conj().T @ phase @ reference[:, first]
        expected.append(np.abs(state) ** 2)
    np.testing.assert_allclose(probabilities.detach().numpy(), expected, atol=1e-13)
    loss = -probabilities[0, 1].log() - probabilities[1, 0].log()
    loss.backward()
    assert coefficients.grad is not None
    for coordinate in [1, 5, 11, 15]:
        losses = []
        for offset in [-1e-6, 1e-6]:
            shifted = coefficients.detach().clone()
            shifted[coordinate] += offset
            changed = quantum_probabilities(
                inputs, torch.matrix_exp(-1j * hermitian(shifted, 4)), Spec("modular", 4, 4)
            )
            losses.append(float(-changed[0, 1].log() - changed[1, 0].log()))
        assert float(coefficients.grad[coordinate]) == pytest.approx(
            (losses[1] - losses[0]) / 2e-6, abs=1e-7
        )


def test_fourier_solution_solves_all_symbol_pairs_without_postselection():
    inputs = torch.cartesian_prod(torch.arange(16), torch.arange(16))
    probabilities = quantum_probabilities(inputs, phase_table(16) / 4, Spec("modular", 16))
    targets = inputs.sum(-1) % 16
    assert metrics(probabilities, targets)["top1"] == 1.0
    torch.testing.assert_close(
        probabilities[torch.arange(256), targets], torch.ones(256, dtype=torch.float64)
    )


def test_dephasing_erases_second_symbol_even_with_retrained_hamiltonian():
    torch.manual_seed(7)
    model = BornModel(Spec("modular", 16), "dephased")
    first = torch.tensor([[3, 0], [3, 4], [3, 15]])
    output = model(first)
    torch.testing.assert_close(output[0], output[1])
    torch.testing.assert_close(output[0], output[2])
    assert metrics(output.detach(), torch.tensor([3, 7, 2]))["norm_error"] < 1e-12
    assert [name for name, _ in model.named_parameters()] == ["coefficients"]
    assert model.coefficients.numel() == 256


def test_image_readout_matches_density_matrix_trace_and_keeps_all_mass():
    spec = Spec("digits", 8, 4)
    rng = np.random.default_rng(23)
    images = rng.uniform(size=(2, 8))
    images /= np.linalg.norm(images, axis=1, keepdims=True)
    inputs = torch.tensor(images)
    unitary = torch.matrix_exp(-1j * hermitian(torch.tensor(rng.normal(size=64)), 8))
    for dephased in [False, True]:
        expected = []
        for image in images:
            density = np.diag(image**2) if dephased else np.outer(image, image)
            u = unitary.numpy()
            expected.append(np.real(np.diag(u @ density @ u.conj().T)).reshape(4, 2).sum(-1))
        actual = quantum_probabilities(inputs, unitary, spec, dephased)
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-13)
        assert metrics(actual, torch.tensor([0, 1]))["norm_error"] < 1e-12
    diagonal = torch.diag(torch.exp(-1j * torch.arange(8, dtype=torch.float64)))
    torch.testing.assert_close(
        quantum_probabilities(inputs, diagonal, spec), inputs.square().reshape(2, 4, 2).sum(-1)
    )


def test_fit_selects_validation_without_a_test_file_and_pairs_initialization(tmp_path):
    inputs = np.array([(a, b) for a in range(16) for b in range(16)])
    np.savez(
        tmp_path / "modular-fit.npz",
        train_inputs=inputs[:160],
        train_targets=inputs[:160].sum(1) % 16,
        validation_inputs=inputs[160:208],
        validation_targets=inputs[160:208].sum(1) % 16,
    )
    hashes = []
    for kind in ["coherent", "dephased"]:
        result = fit(Job(str(tmp_path), str(tmp_path / kind), "modular", kind, 17, 0.03, 3, 160))
        assert result["selected_validation"]["nll"] == min(
            row["validation"]["nll"] for row in result["history"]
        )
        assert result["first_gradient_norm"] > 0
        hashes.append(result["initial_parameter_hashes"])
        assert (
            json.loads((tmp_path / kind / "result.json").read_text())["selected_epoch"]
            == result["selected_epoch"]
        )
    assert hashes[0] == hashes[1]
    with pytest.raises(FileNotFoundError, match="selection_lock"):
        evaluate(tmp_path)


def test_dephased_bound_is_tight_and_coherent_evolution_can_exceed_it():
    powers = np.array([[0.4, 0.3, 0.2, 0.1]])
    assert dephased_nll_bound(np.sqrt(powers), 2) == pytest.approx(-np.log(0.7))
    # Identity and the first rank-two block attain this bound on the diagonal state.
    identity = torch.eye(4, dtype=torch.complex128)
    spec = Spec("digits", 4, 2)
    actual = quantum_probabilities(torch.tensor(np.sqrt(powers)), identity, spec, True)
    assert -float(actual[0, 0].log()) == pytest.approx(dephased_nll_bound(np.sqrt(powers), 2))
    # Uniform amplitudes have dephased maximum 1/2; a Hadamard sends the pure state to |0>.
    amplitudes = torch.full((1, 4), 0.5, dtype=torch.float64)
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / np.sqrt(2)
    unitary = torch.kron(hadamard, hadamard)
    pure = quantum_probabilities(amplitudes, unitary, spec)
    mixed = quantum_probabilities(amplitudes, unitary, spec, True)
    assert float(pure[0, 0]) == pytest.approx(1.0)
    assert float(mixed[0, 0]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="normalized"):
        dephased_nll_bound(np.ones((1, 4)), 2)
