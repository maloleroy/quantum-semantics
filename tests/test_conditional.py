from dataclasses import replace

import pytest
import torch

from src.encoding import pooled_angles
from src.models import AncillaModel, ConditionalModel, ModelConfig
from src.quantum import orthonormal_columns, seed_random


def test_pooling_distinguishes_streams_order_and_padding() -> None:
    table = torch.arange(24, dtype=torch.float64).reshape(1, 3, 8, 1)
    contexts = torch.tensor([[-1, -1, -1, 2], [1, 2, 3, 4], [2, 1, 3, 4]])
    values = pooled_angles(contexts, table, recent=2)[:, 0, 0]
    torch.testing.assert_close(values[0], table[0, 0, 2, 0])
    expected = 4 + (8 + 3) + ((16 + 1) * 0.97 + (16 + 2)) / 1.97
    assert float(values[1]) == pytest.approx(expected)
    assert float(values[1]) != pytest.approx(float(values[2]))
    # No older or middle stream: only the actual most recent word contributes.
    torch.testing.assert_close(
        pooled_angles(contexts[:1, -1:], table, recent=1)[0, 0, 0], values[0]
    )


def test_conditional_matches_dense_context_hamiltonians_and_gradients() -> None:
    torch.set_num_threads(1)
    seed_random(31)
    config = ModelConfig("conditional", 8, 6, rank=3, layers=3, recent_window=2)
    model = ConditionalModel(config)
    contexts = torch.tensor([[-1, -1, -1, -1, -1, 2], [1, 2, 3, 4, 5, 6]])
    inputs = model.states(contexts)
    states = torch.cat((inputs, torch.zeros_like(inputs)), 1)
    angles = pooled_angles(contexts, model.token_angles, config.recent_window)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    identity = torch.eye(2, dtype=torch.complex128)
    local_z: list[torch.Tensor] = []
    for qubit in range(4):
        operator = torch.ones(1, 1, dtype=torch.complex128)
        for bit in reversed(range(4)):
            operator = torch.kron(operator, z if bit == qubit else identity)
        local_z.append(operator)
    for layer in range(config.layers):
        basis = orthonormal_columns(model.factors[layer])
        hamiltonian = (basis * model.energies[layer]) @ basis.mH
        states = states @ torch.matrix_exp(-1j * hamiltonian).T
        if layer < config.layers - 1:
            controls = torch.einsum(
                "bq,qij->bij", angles[:, layer].to(torch.complex128), torch.stack(local_z)
            )
            operators = torch.matrix_exp(-0.5j * controls)
            states = torch.einsum("bij,bj->bi", operators, states)
    expected = states.abs().square().reshape(2, 2, 8).sum(1)
    actual = model(contexts)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    parameters = (model.factors, model.energies, model.token_angles)
    first = torch.autograd.grad(-actual[:, 1].log().mean(), parameters, retain_graph=True)
    second = torch.autograd.grad(-expected[:, 1].log().mean(), parameters)
    for fast, dense in zip(first, second, strict=True):
        assert float(fast.square().sum().sqrt()) > 1e-6
        torch.testing.assert_close(fast, dense, atol=1e-10, rtol=1e-9)


def test_zero_phase_recovers_static_and_freezing_leaves_controller_trainable() -> None:
    config = ModelConfig("conditional", 8, 6, rank=3, phase_scale=0)
    seed_random(19)
    conditional = ConditionalModel(config)
    seed_random(19)
    static = AncillaModel(replace(config, kind="ancilla"))
    contexts = torch.tensor([[-1, -1, -1, -1, -1, 2], [1, 2, 3, 4, 5, 6]])
    torch.testing.assert_close(conditional(contexts), static(contexts))
    frozen = ConditionalModel(replace(config, freeze_core=True, phase_scale=1))
    assert not frozen.factors.requires_grad and not frozen.energies.requires_grad
    assert frozen.token_angles.requires_grad
    fixed = ConditionalModel(replace(config, kind="conditional_fixed", phase_scale=1))
    assert fixed.factors.requires_grad and not fixed.token_angles.requires_grad
