from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.encoding import pooled_angles, product_states
from src.models import AdaptiveMLP, AdaptiveModel, ModelConfig, ProductModel, spectral_controls
from src.quantum import orthonormal_columns, rotate, seed_random, spectral_action
from src.storage import load_weights, save_weights


def test_adaptive_probabilities_and_gradients_match_dense_hamiltonians() -> None:
    torch.set_num_threads(1)
    seed_random(107)
    config = ModelConfig("adaptive", 8, 6, rank=3, layers=3, recent_window=2, phase_scale=0.7)
    model = AdaptiveModel(config)
    with torch.no_grad():
        model.spectral_angles.normal_(std=0.2)
    contexts = torch.tensor([[-1, -1, -1, -1, -1, 2], [1, 2, 3, 4, 5, 6]])
    controls = model.control_offsets + 0.7 * pooled_angles(contexts, model.token_angles, 2)
    shifts = 0.7 * pooled_angles(contexts, model.spectral_angles, 2)[:, 0]
    states = product_states(controls[:, 0])
    for layer in range(config.layers):
        basis = orthonormal_columns(model.factors[layer])
        energies = model.energies[layer] + shifts
        matrices = torch.stack([(basis * energy) @ basis.mH for energy in energies])
        expected = torch.bmm(torch.matrix_exp(-1j * matrices), states.unsqueeze(-1)).squeeze(-1)
        actual = spectral_action(states, model.factors[layer], energies)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual.abs().square().sum(1), torch.ones(2, dtype=torch.float64))
        states = expected
        if layer < config.layers - 1:
            for qubit in range(4):
                states = rotate(states, qubit, controls[:, layer + 1, qubit], "z")
    probabilities = states.abs().square().reshape(2, 2, 8).sum(1)
    actual = model(contexts)
    torch.testing.assert_close(actual, probabilities, atol=1e-12, rtol=1e-12)
    parameters = (
        model.factors,
        model.energies,
        model.token_angles,
        model.control_offsets,
        model.spectral_angles,
    )
    a = torch.autograd.grad(-actual[:, 1].log().mean(), parameters, retain_graph=True)
    b = torch.autograd.grad(-probabilities[:, 1].log().mean(), parameters)
    for fast, dense in zip(a, b, strict=True):
        assert float(fast.square().sum().sqrt()) > 1e-6
        torch.testing.assert_close(fast, dense, atol=1e-10, rtol=1e-9)


def test_adaptive_zero_shift_identity_and_checkpoint(tmp_path: Path) -> None:
    config = ModelConfig("adaptive", 8, 6, rank=3, recent_window=2)
    seed_random(109)
    adaptive = AdaptiveModel(config)
    seed_random(109)
    static = ProductModel(replace(config, kind="product"))
    contexts = torch.tensor([[-1, -1, -1, -1, -1, 2], [1, 2, 3, 4, 5, 6]])
    torch.testing.assert_close(adaptive(contexts), static(contexts), atol=1e-12, rtol=1e-12)
    gradient = torch.autograd.grad(
        -adaptive(contexts)[:, 1].log().mean(), adaptive.spectral_angles
    )[0]
    assert float(gradient.square().sum()) > 1e-6
    with torch.no_grad():
        adaptive.spectral_angles.normal_(std=0.3)
    assert not torch.allclose(adaptive(contexts), static(contexts))
    padded = torch.full((2, 6), -1, dtype=torch.int64)
    torch.testing.assert_close(
        spectral_controls(padded, adaptive.spectral_angles, config),
        torch.zeros(2, 3, dtype=torch.float64),
    )
    path = tmp_path / "adaptive.npz"
    save_weights(path, adaptive)
    restored = AdaptiveModel(config)
    load_weights(path, restored)
    torch.testing.assert_close(restored(contexts), adaptive(contexts))
    constant = AdaptiveModel(replace(config, phase_scale=0))
    with torch.no_grad():
        constant.spectral_angles.normal_(std=0.3)
    torch.testing.assert_close(constant(contexts)[0], constant(contexts)[1])
    with pytest.raises(ValueError, match="Local preparation"):
        ModelConfig("adaptive", 8, 6, decoder="rotated")


def test_adaptive_budgets_and_fixed_basis_control() -> None:
    config = ModelConfig("adaptive", 512, 128, rank=128, recent_window=16)
    model = AdaptiveModel(config)
    classical = AdaptiveMLP(replace(config, kind="adaptive_mlp"))
    capacity = ProductModel(replace(config, kind="product", rank=176))
    assert sum(p.numel() for p in model.parameters()) == 751892
    assert sum(p.numel() for p in classical.parameters()) == 750460
    assert sum(p.numel() for p in capacity.parameters()) == 751988
    fixed = AdaptiveModel(replace(config, freeze_core=True))
    assert sum(p.numel() for p in fixed.parameters() if p.requires_grad) == 227348
    assert not fixed.factors.requires_grad and not fixed.energies.requires_grad
    assert fixed.spectral_angles.requires_grad and fixed.token_angles.requires_grad
    contexts = torch.tensor([[1] * 128, [2] * 128])
    gradient = torch.autograd.grad(-fixed(contexts)[:, 1].log().mean(), fixed.spectral_angles)[0]
    assert float(gradient.square().sum()) > 1e-6
