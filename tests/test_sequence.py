from dataclasses import replace

import torch
from torch import Tensor

from src.encoding import bloch_coordinates, factor_product, pooled_angles, sequential_factors
from src.models import ModelConfig, SequenceMLP, SequenceModel
from src.quantum import orthonormal_columns, rotate, seed_random


def explicit_preparation(model: SequenceModel, contexts: Tensor) -> Tensor:
    count = model.sequence_angles.shape[-1]
    states = torch.zeros(len(contexts), 2**count, dtype=torch.complex128)
    states[:, 0] = 1
    for qubit in range(count):
        states = rotate(states, qubit, model.control_offsets[0, qubit], "y")
        states = rotate(states, qubit, model.control_offsets[1, qubit], "z")
    angles = model.sequence_angles[:, contexts.clamp_min(0)]
    ages = torch.arange(contexts.shape[1] - 1, -1, -1, dtype=torch.float64)
    angles = angles * ((contexts >= 0) * 0.97**ages * model.config.phase_scale)[None, :, :, None]
    if model.config.kind == "sequence_pooled":
        angles = angles.sum(2, keepdim=True)
    for position in range(angles.shape[2]):
        for qubit in range(count):
            states = rotate(states, qubit, angles[0, :, position, qubit], "y")
            states = rotate(states, qubit, angles[1, :, position, qubit], "z")
    return states


def test_sequence_probabilities_and_gradients_match_explicit_dense_circuit() -> None:
    torch.set_num_threads(1)
    for kind in ("sequence", "sequence_pooled"):
        seed_random(73)
        config = ModelConfig(kind, 8, 6, rank=3, layers=3, recent_window=2, phase_scale=0.8)
        model = SequenceModel(config)
        contexts = torch.tensor([[-1, -1, -1, -1, -1, 2], [1, 2, 3, 4, 5, 6]])
        states = explicit_preparation(model, contexts)
        torch.testing.assert_close(model.states(contexts), states, atol=1e-12, rtol=1e-12)
        reduced = states.reshape(2, 2, 8) @ states.reshape(2, 2, 8).mH
        torch.testing.assert_close(
            (reduced @ reduced).diagonal(dim1=-2, dim2=-1).sum(1).real,
            torch.ones(2, dtype=torch.float64),
        )
        controls = model.control_offsets[2:] + config.phase_scale * pooled_angles(
            contexts, model.token_angles, config.recent_window
        )
        for layer in range(config.layers):
            basis = orthonormal_columns(model.factors[layer])
            hamiltonian = (basis * model.energies[layer]) @ basis.mH
            states = states @ torch.matrix_exp(-1j * hamiltonian).T
            if layer < config.layers - 1:
                for qubit in range(4):
                    states = rotate(states, qubit, controls[:, layer, qubit], "z")
        expected = states.abs().square().reshape(2, 2, 8).sum(1)
        actual = model(contexts)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        parameters = (
            model.factors,
            model.energies,
            model.sequence_angles,
            model.token_angles,
            model.control_offsets,
        )
        first = torch.autograd.grad(-actual[:, 1].log().mean(), parameters, retain_graph=True)
        second = torch.autograd.grad(-expected[:, 1].log().mean(), parameters)
        for fast, dense in zip(first, second, strict=True):
            assert float(fast.square().sum().sqrt()) > 1e-6
            torch.testing.assert_close(fast, dense, atol=1e-10, rtol=1e-9)


def test_sequence_order_padding_and_physical_classical_features() -> None:
    table = torch.tensor([[[0.8], [-0.5]], [[0.2], [1.1]]], dtype=torch.float64)
    offsets = torch.zeros(2, 1, dtype=torch.float64)
    contexts = torch.tensor([[0, 1], [1, 0]])
    ordered = factor_product(*sequential_factors(contexts, table, offsets, decay=1))
    pooled = factor_product(*sequential_factors(contexts, table, offsets, decay=1, pooled=True))
    assert not torch.allclose(ordered[0].abs().square(), ordered[1].abs().square())
    torch.testing.assert_close(pooled[0], pooled[1])
    zero, one = sequential_factors(contexts, table, offsets)
    padded = torch.cat((torch.full((2, 3), -1), contexts), 1)
    for actual, expected in zip(
        sequential_factors(padded, table, offsets), (zero, one), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    bloch = bloch_coordinates(zero, one)
    phase = torch.exp(1j * torch.tensor([[0.4], [-0.3]], dtype=torch.float64))
    torch.testing.assert_close(bloch_coordinates(zero * phase, one * phase), bloch)
    density = torch.stack((zero, one), -1).unsqueeze(-1) * torch.stack(
        (zero, one), -1
    ).conj().unsqueeze(-2)
    reconstructed = torch.stack(
        (
            torch.stack(((1 + bloch[..., 2]) / 2, (bloch[..., 0] - 1j * bloch[..., 1]) / 2), -1),
            torch.stack(((bloch[..., 0] + 1j * bloch[..., 1]) / 2, (1 - bloch[..., 2]) / 2), -1),
        ),
        -2,
    )
    torch.testing.assert_close(density, reconstructed)


def test_sequence_controls_and_classical_budget() -> None:
    config = ModelConfig("sequence", 512, 128, rank=64, recent_window=16)
    model = SequenceModel(config)
    classical = SequenceMLP(replace(config, kind="sequence_mlp"))
    assert sum(p.numel() for p in model.parameters()) == 287902
    assert sum(p.numel() for p in classical.parameters()) == 287711
    fixed = SequenceModel(replace(config, kind="sequence_fixed"))
    assert sum(p.numel() for p in fixed.parameters() if p.requires_grad) == 262272
    frozen = SequenceModel(replace(config, freeze_core=True))
    assert sum(p.numel() for p in frozen.parameters() if p.requires_grad) == 25630
    assert not frozen.factors.requires_grad and not frozen.energies.requires_grad
    constant = SequenceModel(ModelConfig("sequence", 8, 6, rank=3, phase_scale=0))
    contexts = torch.tensor([[-1, -1, -1, -1, -1, 2], [1, 2, 3, 4, 5, 6]])
    predictions = constant(contexts)
    torch.testing.assert_close(predictions[0], predictions[1])
