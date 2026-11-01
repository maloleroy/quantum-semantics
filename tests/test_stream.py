from dataclasses import replace

import pytest
import torch

from src.encoding import pooled_angles
from src.models import ModelConfig, StreamMLP, StreamModel, stream_inputs
from src.quantum import orthonormal_columns, rotate, seed_random


def test_stream_probabilities_and_gradients_match_dense_block_circuit() -> None:
    torch.set_num_threads(1)
    for kind in ("stream", "stream_pooled"):
        seed_random(101)
        config = ModelConfig(kind, 8, 8, rank=3, layers=4, recent_window=2)
        model = StreamModel(config)
        contexts = torch.tensor([[-1, -1, -1, -1, -1, -1, -1, 2], [1, 2, 3, 4, 5, 6, 7, 1]])
        blocks = (
            (contexts,) * 4
            if kind == "stream_pooled"
            else tuple(contexts[:, start : start + 2] for start in range(0, 8, 2))
        )
        states = torch.zeros(2, 16, dtype=torch.complex128)
        states[:, 0] = 1
        for stage, block in enumerate(blocks):
            slot = min(stage, 1)
            angles = (
                model.control_offsets[stage]
                + pooled_angles(block, model.token_angles[slot : slot + 1], 2)[:, 0]
            )
            for qubit in range(4):
                states = rotate(states, qubit, angles[:, qubit], "y" if stage == 0 else "z")
            basis = orthonormal_columns(model.factors[stage])
            matrix = (basis * model.energies[stage]) @ basis.mH
            states = states @ torch.matrix_exp(-1j * matrix).T
        expected = states.abs().square().reshape(2, 2, 8).sum(1)
        actual = model(contexts)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        parameters = (model.factors, model.energies, model.token_angles, model.control_offsets)
        a = torch.autograd.grad(-actual[:, 1].log().mean(), parameters, retain_graph=True)
        b = torch.autograd.grad(-expected[:, 1].log().mean(), parameters)
        for fast, dense in zip(a, b, strict=True):
            assert float(fast.square().sum().sqrt()) > 1e-6
            torch.testing.assert_close(fast, dense, atol=1e-10, rtol=1e-9)


def test_stream_later_blocks_act_after_preparation_and_padding_is_empty() -> None:
    seed_random(103)
    config = ModelConfig("stream", 8, 8, rank=3, layers=4, recent_window=2)
    model = StreamModel(config)
    contexts = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 1], [1, 2, 3, 4, 5, 6, 2, 3]])
    initial, controls = stream_inputs(contexts, model.token_angles, model.control_offsets, config)
    torch.testing.assert_close(initial[0], initial[1])
    torch.testing.assert_close(controls[0, :2], controls[1, :2])
    assert not torch.allclose(controls[0, 2], controls[1, 2])
    assert not torch.allclose(model(contexts)[0], model(contexts)[1])
    padded = torch.tensor([[-1, -1, -1, -1, -1, -1, 2, 3]])
    initial, controls = stream_inputs(padded, model.token_angles, model.control_offsets, config)
    torch.testing.assert_close(initial, torch.full((1, 16), 0.25, dtype=torch.complex128))
    torch.testing.assert_close(controls[:, :2], torch.zeros(1, 2, 4, dtype=torch.float64))
    constant = StreamModel(replace(config, phase_scale=0))
    torch.testing.assert_close(constant(contexts)[0], constant(contexts)[1])
    with pytest.raises(ValueError, match="divide evenly"):
        ModelConfig("stream", 8, 7, layers=4)


def test_stream_parameter_budgets_and_frozen_dynamics() -> None:
    config = ModelConfig("stream", 512, 128, rank=32, layers=4, recent_window=16)
    model = StreamModel(config)
    classical = StreamMLP(replace(config, kind="stream_mlp"))
    assert sum(p.numel() for p in model.parameters()) == 293032
    assert sum(p.numel() for p in classical.parameters()) == 292961
    pooled = StreamModel(replace(config, kind="stream_pooled"))
    assert sum(p.numel() for p in pooled.parameters()) == 293032
    frozen = StreamModel(replace(config, freeze_core=True))
    assert sum(p.numel() for p in frozen.parameters() if p.requires_grad) == 30760
    assert not frozen.factors.requires_grad and not frozen.energies.requires_grad
    assert frozen.token_angles.requires_grad and frozen.control_offsets.requires_grad
