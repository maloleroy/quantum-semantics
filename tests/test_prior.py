from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import Dataset, bigram_probabilities, prepare
from src.encoding import pooled_angles
from src.models import ModelConfig, ModelKind, PriorMLP, PriorModel, build_model
from src.quantum import orthonormal_columns, rotate, seed_random
from src.storage import load_weights, save_weights
from src.training import TrainConfig, selected_models, train


@pytest.fixture
def prior_data(tmp_path: Path) -> Dataset:
    return prepare(
        Path("data/phrases.csv"),
        tmp_path,
        vocabulary=8,
        window=6,
        train_limit=24,
        validation_limit=8,
        test_limit=8,
    )


def test_prior_initialization_padding_and_train_only_counts(prior_data: Dataset) -> None:
    torch.set_num_threads(1)
    contexts = torch.tensor([[1, 2, 3, 4, 5, 6], [-1, -1, -1, -1, -1, -1]])
    expected = torch.tensor(
        np.stack(
            (
                bigram_probabilities(prior_data.transitions, prior_data.unigram)[6],
                prior_data.unigram,
            )
        )
    )
    for kind in ("prior", "prior_mlp", "prior_unigram", "prior_position", "prior_position_mlp"):
        config = ModelConfig(kind, 8, 6, rank=3, recent_window=2)
        model = build_model(config, prior_data, 11)
        target = (
            expected if kind != "prior_unigram" else torch.tensor(prior_data.unigram).expand(2, -1)
        )
        torch.testing.assert_close(model(contexts), target, atol=1e-12, rtol=1e-12)
        changed = replace(prior_data, validation=prior_data.test, test=prior_data.validation)
        untouched = build_model(config, changed, 11)
        torch.testing.assert_close(untouched(contexts), model(contexts), atol=0, rtol=0)
    with pytest.raises(ValueError, match="Prior models"):
        ModelConfig("prior", 8, 6, decoder="rotated")
    with pytest.raises(ValueError, match="Freezing Hamiltonian"):
        ModelConfig("prior_mlp", 8, 6, freeze_core=True)


@pytest.mark.parametrize("kind", ["prior", "prior_position"])
def test_prior_matches_dense_context_hamiltonians(prior_data: Dataset, kind: ModelKind) -> None:
    seed_random(223)
    config = ModelConfig(kind, 8, 6, rank=3, layers=3, recent_window=3, phase_scale=0.7)
    model = PriorModel(config, prior_data)
    with torch.no_grad():
        model.energies.normal_(std=0.3)
        model.spectral_angles.normal_(std=0.2)
        if model.position_angles is not None and model.position_spectra is not None:
            model.position_angles.normal_(std=0.3)
            model.position_spectra.normal_(std=0.3)
    contexts = torch.tensor([[-1, -1, -1, -1, -1, 2], [1, 2, 3, 4, 5, 6]])
    states = torch.cat((model.states(contexts), torch.zeros(2, 8, dtype=torch.complex128)), 1)
    angles = 0.7 * pooled_angles(contexts, model.token_angles, 3, model.position_angles)
    shifts = 0.7 * pooled_angles(contexts, model.spectral_angles, 3, model.position_spectra)[:, 0]
    for layer in range(config.layers):
        basis = orthonormal_columns(model.factors[layer])
        matrices = torch.stack(
            [(basis * energy) @ basis.mH for energy in model.energies[layer] + shifts]
        )
        states = torch.bmm(torch.matrix_exp(-1j * matrices), states.unsqueeze(-1)).squeeze(-1)
        if layer < config.layers - 1:
            for qubit in range(4):
                states = rotate(states, qubit, angles[:, layer, qubit], "z")
    expected = states.abs().square().reshape(2, 2, 8).sum(1)
    actual = model(contexts)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual.sum(1), torch.ones(2, dtype=torch.float64))
    parameters = (model.factors, model.energies, model.token_angles, model.spectral_angles)
    if model.position_angles is not None and model.position_spectra is not None:
        parameters += (model.position_angles, model.position_spectra)
    fast = torch.autograd.grad(-actual[:, 1].log().mean(), parameters, retain_graph=True)
    dense = torch.autograd.grad(-expected[:, 1].log().mean(), parameters)
    for a, b in zip(fast, dense, strict=True):
        assert float(a.abs().sum()) > 1e-7
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("kind", ["prior", "prior_position"])
def test_prior_identity_can_learn_and_freeze_is_explicit(
    prior_data: Dataset, tmp_path: Path, kind: ModelKind
) -> None:
    config = ModelConfig(kind, 8, 6, rank=3, recent_window=3)
    contexts = torch.tensor(prior_data.train.contexts[:8])
    targets = torch.tensor(prior_data.train.targets[:8])
    for freeze in (False, True):
        model = PriorModel(replace(config, freeze_core=freeze), prior_data)
        initial = model(contexts).detach()
        factors = model.factors.detach().clone()
        for _ in range(3):
            model.zero_grad()
            loss = -model(contexts)[torch.arange(8), targets].log().mean()
            loss.backward()
            assert model.spectral_angles.grad is not None
            assert float(model.spectral_angles.grad.abs().sum()) > 0
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad, alpha=-0.01)
        assert not torch.allclose(model(contexts), initial)
        assert torch.equal(model.factors, factors) == freeze
        assert model.factors.requires_grad != freeze and model.energies.requires_grad != freeze
        assert model.spectral_angles.requires_grad and model.token_angles.requires_grad
        path = tmp_path / f"prior-{freeze}.npz"
        save_weights(path, model)
        restored = PriorModel(replace(config, freeze_core=freeze), prior_data)
        load_weights(path, restored)
        torch.testing.assert_close(restored(contexts), model(contexts), atol=0, rtol=0)
    classical = PriorMLP(
        replace(config, kind="prior_position_mlp" if kind == "prior_position" else "prior_mlp"),
        prior_data,
    )
    quantum = PriorModel(config, prior_data)
    assert sum(p.numel() for p in classical.parameters()) <= sum(
        p.numel() for p in quantum.parameters()
    )


@pytest.mark.parametrize("kind", ["prior", "prior_position", "prior_position_mlp"])
def test_prior_locked_fit_uses_new_kind_and_no_test(
    prior_data: Dataset, tmp_path: Path, kind: ModelKind
) -> None:
    dataset_path = tmp_path / "dataset.npz"
    output = train(
        dataset_path,
        [ModelConfig(kind, 8, 6, rank=3, recent_window=3)],
        TrainConfig(epochs=2, batch_size=8, selection="nll"),
        [11],
        tmp_path / "runs",
    )
    dataset, records = selected_models(output)
    assert len(records) == 1 and records[0]["model"]["kind"] == kind
    assert dataset.size == prior_data.size and not (output / "test.csv").exists()


@pytest.mark.parametrize("recent", [1, 3, 6])
def test_learned_positions_match_scalar_reference_and_ignore_padding(recent: int) -> None:
    torch.set_num_threads(1)
    seed_random(31)
    contexts = torch.tensor([[-1, -1, -1, -1, -1, -1], [-1, -1, 2, 1, 3, 4], [1, 2, 3, 4, 5, 6]])
    table = torch.randn(2, 3, 8, 3, dtype=torch.float64, requires_grad=True)
    logits = torch.zeros(2, 5, 3, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(
        pooled_angles(contexts, table, recent, logits),
        pooled_angles(contexts, table, recent),
        atol=1e-14,
        rtol=1e-14,
    )
    with torch.no_grad():
        logits.normal_(std=2)
    actual = pooled_angles(contexts, table, recent, logits)
    expected = torch.zeros_like(actual)
    for row in range(len(contexts)):
        for stage in range(2):
            for feature in range(3):
                for stream, (start, stop, decay) in enumerate(
                    ((5, 6, 1.0), (6 - recent, 5, 0.7), (0, 6 - recent, 0.97))
                ):
                    positions = [p for p in range(start, stop) if int(contexts[row, p]) >= 0]
                    if positions:
                        weights = torch.stack(
                            [
                                table.new_tensor(decay ** (stop - p - 1))
                                * (logits[stage, p, feature].exp() if stream else 1)
                                for p in positions
                            ]
                        )
                        values = torch.stack(
                            [table[stage, stream, contexts[row, p], feature] for p in positions]
                        )
                        expected[row, stage, feature] += (weights * values).sum() / weights.sum()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    fast = torch.autograd.grad(actual.square().sum(), (table, logits), retain_graph=True)
    slow = torch.autograd.grad(expected.square().sum(), (table, logits))
    for a, b in zip(fast, slow, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
    assert torch.equal(actual[0], torch.zeros_like(actual[0]))
    # An all-padding context must stay finite even when a padded lag gets huge weight.
    with torch.no_grad():
        logits[:, 0] = 1000
    assert torch.isfinite(pooled_angles(contexts, table, recent, logits)).all()
