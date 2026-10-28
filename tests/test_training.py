from pathlib import Path

import pytest
import torch

from src.data import Dataset, prepare
from src.metrics import probability_scores, summarize
from src.models import MODEL_KINDS, ModelConfig, ModelKind, build_model
from src.storage import load_weights, save_weights
from src.training import TrainConfig, selected_models, train
from src.training import test as score_test


def tiny_data(path: Path) -> Dataset:
    source = path / "sentences.csv"
    source.write_text(
        "\n".join(
            f"{a} sees the {b}"
            for a in ("red", "blue", "green", "white")
            for b in ("bird", "cat", "fish", "tree")
        )
    )
    return prepare(source, path, vocabulary=8, train_limit=24, validation_limit=8, test_limit=8)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_model_probabilities_and_portable_weights(kind: ModelKind, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    dataset = tiny_data(tmp_path)
    config = ModelConfig(kind, dataset.size, dataset.window, latent=4)
    model = build_model(config, dataset, 11)
    context = torch.tensor(dataset.train.contexts[:4])
    expected = model(context).detach()
    torch.testing.assert_close(expected.sum(1), torch.ones(len(context), dtype=torch.float64))
    save_weights(tmp_path / "weights.npz", model)
    restored = build_model(config, dataset, 22)
    load_weights(tmp_path / "weights.npz", restored)
    torch.testing.assert_close(restored(context), expected)


def test_sampling_success_is_not_argmax_accuracy() -> None:
    probabilities = torch.tensor([[0.6, 0.4], [0.9, 0.1]], dtype=torch.float64)
    measured = summarize(probability_scores(probabilities, torch.tensor([0, 1])))
    assert measured.success == pytest.approx(0.35)
    assert measured.accuracy == pytest.approx(0.5)
    assert measured.known_success == pytest.approx(0.1)


def test_validation_selection_and_separate_locked_test(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    dataset = tiny_data(tmp_path)
    configs = [ModelConfig(kind, dataset.size, dataset.window) for kind in ("hamiltonian", "mlp")]
    run = train(
        tmp_path / "dataset.npz", configs, TrainConfig(epochs=2, batch_size=16), [11], tmp_path
    )
    assert not (run / "test.csv").exists()
    _, records = selected_models(run)
    assert len(records) == 2
    assert score_test(run).exists()
    with pytest.raises(ValueError, match="already scored"):
        score_test(run)


def test_interleaving_matches_explicit_local_gates_and_has_gradients(tmp_path: Path) -> None:
    from src.models import InterleavedModel
    from src.quantum import evolve, hermitian, rotate

    torch.set_num_threads(1)
    dataset = tiny_data(tmp_path)
    model = InterleavedModel(ModelConfig("interleaved", dataset.size, dataset.window))
    contexts = torch.tensor(dataset.train.contexts[:4])
    states = model.states(contexts) @ evolve(hermitian(model.coefficients[0], dataset.size)).T
    valid = contexts >= 0
    features = model.token_angles[contexts.clamp_min(0)] * valid[:, :, None]
    angles = (features * model.position_signs[0]).sum(1) / valid.sum(1).clamp_min(1).sqrt()[:, None]
    for q in range(angles.shape[1]):
        states = rotate(states, q, angles[:, q], "z")
    expected = (states @ evolve(hermitian(model.coefficients[1], dataset.size)).T).abs().square()
    actual = model(contexts)
    torch.testing.assert_close(actual, expected)
    loss = -actual[:, 1].log().mean()
    loss.backward()
    gradient = model.coefficients.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert (gradient.square().sum(dim=1).sqrt() > 1e-6).all()
    # Finite difference checks the combined phase/evolution computational path.
    epsilon = 1e-5
    with torch.no_grad():
        model.coefficients[0, 3] += epsilon
        plus = -model(contexts)[:, 1].log().mean()
        model.coefficients[0, 3] -= 2 * epsilon
        minus = -model(contexts)[:, 1].log().mean()
        model.coefficients[0, 3] += epsilon
    assert float((plus - minus) / (2 * epsilon)) == pytest.approx(float(gradient[0, 3]), abs=1e-6)


@pytest.mark.parametrize("kind", ["ancilla", "ancilla_dephased"])
def test_ancilla_readout_matches_density_matrix(kind: ModelKind, tmp_path: Path) -> None:
    from src.models import AncillaModel
    from src.quantum import orthonormal_columns

    dataset = tiny_data(tmp_path)
    config = ModelConfig(kind, dataset.size, dataset.window, rank=3, ancilla_size=2)
    model = AncillaModel(config)
    contexts = torch.tensor(dataset.train.contexts[:3])
    states = model.states(contexts)
    operator = torch.eye(model.dimension, dtype=torch.complex128)
    for factors, energies in zip(model.factors, model.energies, strict=True):
        basis = orthonormal_columns(factors)
        operator = torch.matrix_exp(-1j * ((basis * energies) @ basis.mH)) @ operator
    expected: list[torch.Tensor] = []
    for state in states:
        input_density = state[:, None] * state.conj()[None, :]
        if kind == "ancilla_dephased":
            input_density = torch.diag(state.abs().square()).to(torch.complex128)
        isometry = operator[:, : dataset.size]
        density = isometry @ input_density @ isometry.mH
        expected.append(density.diagonal().real.reshape(2, dataset.size).sum(0))
    actual = model(contexts)
    dense = torch.stack(expected)
    torch.testing.assert_close(actual, dense, atol=1e-12, rtol=1e-12)
    first = torch.autograd.grad(-actual[:, 1].log().mean(), model.factors, retain_graph=True)[0]
    second = torch.autograd.grad(-dense[:, 1].log().mean(), model.factors)[0]
    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-9)


def test_single_auxiliary_state_recovers_word_only_model(tmp_path: Path) -> None:
    dataset = tiny_data(tmp_path)
    contexts = torch.tensor(dataset.train.contexts[:3])
    plain = build_model(
        ModelConfig("lowrank_static", dataset.size, dataset.window, rank=3), dataset, 19
    )
    auxiliary = build_model(
        ModelConfig("ancilla", dataset.size, dataset.window, rank=3, ancilla_size=1), dataset, 19
    )
    torch.testing.assert_close(plain(contexts), auxiliary(contexts))
