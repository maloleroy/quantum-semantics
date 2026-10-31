from dataclasses import replace
from pathlib import Path

import torch

from src.encoding import pooled_angles, product_states
from src.models import ModelConfig, ProductMLP, ProductModel
from src.quantum import orthonormal_columns, rotate, seed_random


def test_product_preparation_matches_local_gates_and_is_separable() -> None:
    angles = torch.tensor([[0.2, -0.7, 1.1], [2.0, 0.3, -1.2]], dtype=torch.float64)
    states = torch.zeros(2, 8, dtype=torch.complex128)
    states[:, 0] = 1
    for qubit in range(3):
        states = rotate(states, qubit, angles[:, qubit], "y")
    actual = product_states(angles)
    torch.testing.assert_close(actual, states)
    split = actual.reshape(2, 2, 4)
    reduced = split @ split.mH
    torch.testing.assert_close(
        (reduced @ reduced).diagonal(dim1=-2, dim2=-1).sum(1).real,
        torch.ones(2, dtype=torch.float64),
    )


def test_product_dynamics_and_gradients_match_dense_rotations() -> None:
    torch.set_num_threads(1)
    seed_random(61)
    config = ModelConfig("product", 8, 6, rank=3, layers=3, recent_window=2)
    model = ProductModel(config)
    contexts = torch.tensor([[-1, -1, -1, -1, -1, 2], [1, 2, 3, 4, 5, 6]])
    angles = model.control_offsets + pooled_angles(
        contexts, model.token_angles, config.recent_window
    )
    states = torch.zeros(2, 16, dtype=torch.complex128)
    states[:, 0] = 1
    for qubit in range(4):
        states = rotate(states, qubit, angles[:, 0, qubit], "y")
    for layer in range(config.layers):
        basis = orthonormal_columns(model.factors[layer])
        hamiltonian = (basis * model.energies[layer]) @ basis.mH
        states = states @ torch.matrix_exp(-1j * hamiltonian).T
        if layer < config.layers - 1:
            for qubit in range(4):
                states = rotate(states, qubit, angles[:, layer + 1, qubit], "z")
    expected = states.abs().square().reshape(2, 2, 8).sum(1)
    actual = model(contexts)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    parameters = (model.factors, model.energies, model.token_angles, model.control_offsets)
    first = torch.autograd.grad(-actual[:, 1].log().mean(), parameters, retain_graph=True)
    second = torch.autograd.grad(-expected[:, 1].log().mean(), parameters)
    for fast, dense in zip(first, second, strict=True):
        assert float(fast.square().sum().sqrt()) > 1e-6
        torch.testing.assert_close(fast, dense, atol=1e-10, rtol=1e-9)


def test_product_controls_and_classical_budget() -> None:
    config = ModelConfig("product", 512, 128, rank=64, recent_window=16)
    model = ProductModel(config)
    reference = ProductMLP(replace(config, kind="product_mlp"))
    assert sum(p.numel() for p in model.parameters()) == 293012
    assert sum(p.numel() for p in reference.parameters()) == 292695
    frozen = ProductModel(replace(config, freeze_core=True))
    assert not frozen.factors.requires_grad and not frozen.energies.requires_grad
    assert frozen.token_angles.requires_grad and frozen.control_offsets.requires_grad
    fixed = ProductModel(replace(config, kind="product_fixed"))
    assert fixed.factors.requires_grad and fixed.energies.requires_grad
    assert not fixed.token_angles.requires_grad and not fixed.control_offsets.requires_grad
    constant = ProductModel(ModelConfig("product", 8, 6, rank=3, phase_scale=0))
    contexts = torch.tensor([[-1, -1, -1, -1, -1, 2], [1, 2, 3, 4, 5, 6]])
    predictions = constant(contexts)
    torch.testing.assert_close(predictions[0], predictions[1])


def test_rotated_product_matches_explicit_word_gates_and_gradients() -> None:

    torch.set_num_threads(1)
    seed_random(83)
    config = ModelConfig("product", 8, 6, rank=3, layers=3, recent_window=2, decoder="rotated")
    model = ProductModel(config)
    assert model.readout_angles is not None
    with torch.no_grad():
        model.readout_angles.copy_(torch.randn(3, 2, dtype=torch.float64) * 0.3)
    contexts = torch.tensor([[-1, -1, -1, -1, -1, 2], [1, 2, 3, 4, 5, 6]])
    controls = model.control_offsets + pooled_angles(contexts, model.token_angles, 2)
    states = product_states(controls[:, 0])
    for layer in range(config.layers):
        basis = orthonormal_columns(model.factors[layer])
        states = states @ torch.matrix_exp(-1j * ((basis * model.energies[layer]) @ basis.mH)).T
        if layer < config.layers - 1:
            for qubit in range(4):
                states = rotate(states, qubit, controls[:, layer + 1, qubit], "z")
    for qubit in range(3):
        states = rotate(states, qubit, model.readout_angles[qubit, 0], "x")
        states = rotate(states, qubit, model.readout_angles[qubit, 1], "y")
    expected = states.abs().square().reshape(2, 2, 8).sum(1)
    actual = model(contexts)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    parameters = (
        model.factors,
        model.energies,
        model.token_angles,
        model.control_offsets,
        model.readout_angles,
    )
    first = torch.autograd.grad(-actual[:, 1].log().mean(), parameters, retain_graph=True)
    second = torch.autograd.grad(-expected[:, 1].log().mean(), parameters)
    for fast, dense in zip(first, second, strict=True):
        assert float(fast.square().sum().sqrt()) > 1e-6
        torch.testing.assert_close(fast, dense, atol=1e-10, rtol=1e-9)
    auxiliary_rotated = rotate(states, 3, torch.tensor(0.83, dtype=torch.float64), "y")
    torch.testing.assert_close(auxiliary_rotated.abs().square().reshape(2, 2, 8).sum(1), expected)


def test_rotated_product_initial_identity_budgets_and_checkpoint(tmp_path: Path) -> None:
    from src.storage import load_weights, save_weights

    config = ModelConfig("product", 8, 6, rank=3, recent_window=2)
    seed_random(97)
    plain = ProductModel(config)
    seed_random(97)
    rotated = ProductModel(replace(config, decoder="rotated"))
    contexts = torch.tensor([[-1, -1, -1, -1, -1, 2], [1, 2, 3, 4, 5, 6]])
    torch.testing.assert_close(plain(contexts), rotated(contexts), atol=1e-12, rtol=1e-12)
    assert rotated.readout_angles is not None
    with torch.no_grad():
        rotated.readout_angles.fill_(0.2)
    path = tmp_path / "rotated.npz"
    save_weights(path, rotated)
    restored = ProductModel(replace(config, decoder="rotated"))
    load_weights(path, restored)
    torch.testing.assert_close(rotated(contexts), restored(contexts))
    large = ModelConfig("product", 512, 128, rank=128, recent_window=16, decoder="rotated")
    model = ProductModel(large)
    assert sum(p.numel() for p in model.parameters()) == 555302
    frozen = ProductModel(replace(large, freeze_core=True))
    assert sum(p.numel() for p in frozen.parameters() if p.requires_grad) == 30758
    assert frozen.readout_angles is not None
    assert frozen.readout_angles.requires_grad and not frozen.factors.requires_grad
    fixed = ProductModel(replace(large, kind="product_fixed"))
    assert sum(p.numel() for p in fixed.parameters() if p.requires_grad) == 524562
    assert fixed.readout_angles is not None
    assert fixed.readout_angles.requires_grad and not fixed.token_angles.requires_grad
