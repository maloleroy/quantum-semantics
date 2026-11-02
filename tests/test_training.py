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


def test_configured_decay_preserves_distant_token_population(tmp_path: Path) -> None:
    from src.encoding import amplitudes
    from src.models import AncillaModel

    dataset = tiny_data(tmp_path)
    contexts = torch.tensor([[-1, 1, 2, 3]])
    model = AncillaModel(ModelConfig("ancilla", dataset.size, 4, amplitude_decay=0.97))
    actual = model.states(contexts)
    torch.testing.assert_close(actual, amplitudes(contexts, dataset.size, decay=0.97))
    assert float(actual[0, 1].abs().square() / actual[0, 3].abs().square()) == pytest.approx(
        0.97**2
    )
    assert float(actual[0, 1].abs().square()) > float(
        amplitudes(contexts, dataset.size)[0, 1].abs().square()
    )
    for invalid in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="Amplitude decay"):
            ModelConfig("ancilla", dataset.size, 4, amplitude_decay=invalid)


@pytest.mark.parametrize("kind", ["segmented", "segmented_mixture"])
def test_segmented_input_and_sector_dephasing(kind: ModelKind) -> None:
    from src.encoding import amplitudes
    from src.models import SegmentedModel
    from src.quantum import orthonormal_columns

    torch.set_num_threads(1)
    model = SegmentedModel(ModelConfig(kind, 8, 4, rank=3, recent_window=2))
    contexts = torch.tensor([[1, 2, 3, 4], [-1, -1, 2, 3]])
    states = model.states(contexts)
    expected = (
        torch.cat(
            (amplitudes(contexts[:1, -2:], 8), amplitudes(contexts[:1, :2], 8, decay=0.97)), 1
        )
        / 2**0.5
    )
    torch.testing.assert_close(states[:1], expected)
    torch.testing.assert_close(states[1, :8], amplitudes(contexts[1:, -2:], 8)[0])
    assert torch.count_nonzero(states[1, 8:]) == 0
    operator = torch.eye(16, dtype=torch.complex128)
    for factors, energies in zip(model.factors, model.energies, strict=True):
        basis = orthonormal_columns(factors)
        operator = torch.matrix_exp(-1j * ((basis * energies) @ basis.mH)) @ operator
    density = states[:, :, None] * states.conj()[:, None, :]
    if kind == "segmented_mixture":
        sectors = torch.arange(16) // 8
        density = density * (sectors[:, None] == sectors[None, :])
    evolved = operator @ density @ operator.mH
    reference = evolved.diagonal(dim1=1, dim2=2).real.reshape(2, 2, 8).sum(1)
    actual = model(contexts)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    first = torch.autograd.grad(-actual[:, 1].log().mean(), model.factors, retain_graph=True)[0]
    second = torch.autograd.grad(-reference[:, 1].log().mean(), model.factors)[0]
    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-9)


def test_learning_rate_schedule_endpoints_and_validation() -> None:
    from dataclasses import replace

    from src.training import epoch_learning_rate

    config = TrainConfig(epochs=3, learning_rate=0.003, lr_schedule="cosine", min_lr_ratio=0.1)
    assert [epoch_learning_rate(config, epoch) for epoch in range(3)] == pytest.approx(
        [0.003, 0.00165, 0.0003]
    )
    constant = replace(config, lr_schedule="constant")
    assert [epoch_learning_rate(constant, epoch) for epoch in range(3)] == [0.003] * 3
    assert epoch_learning_rate(replace(config, epochs=1), 0) == 0.003
    for invalid in (0, -1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="minimum LR ratio"):
            replace(config, min_lr_ratio=invalid)
    for epoch in (-1, 3):
        with pytest.raises(ValueError, match="outside"):
            epoch_learning_rate(config, epoch)


def test_cosine_fit_changes_updates_and_preserves_selection_lock(tmp_path: Path) -> None:
    import csv
    from dataclasses import replace

    import numpy as np

    from src.storage import read_json
    from src.training import fit

    torch.set_num_threads(1)
    dataset = tiny_data(tmp_path)
    model = ModelConfig("mlp", dataset.size, dataset.window)
    config = TrainConfig(epochs=3, batch_size=16, learning_rate=0.01, selection="nll")
    histories: dict[str, list[dict[str, str]]] = {}
    for name, setting in (
        ("constant", config),
        ("identity", replace(config, lr_schedule="cosine", min_lr_ratio=1)),
        ("decay", replace(config, lr_schedule="cosine")),
    ):
        root = tmp_path / name
        record = fit(dataset, model, setting, 13, root)
        with (root / "history.csv").open() as file:
            history = list(csv.DictReader(file))
        histories[name] = history
        best = min(history, key=lambda row: (float(row["nll"]), -float(row["success"])))
        assert int(best["epoch"]) == record["selected_epoch"]
        assert float(best["nll"]) == pytest.approx(record["validation"]["nll"])
        assert history[-1]["next_epoch_learning_rate"] == ""
        assert not (root / "test.csv").exists()
        metadata = read_json(root / "fit.json")
        assert metadata["training"] == {
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 0.01,
            "selection": "nll",
            "loss": "nll",
            "lr_schedule": setting.lr_schedule,
            "min_lr_ratio": setting.min_lr_ratio,
        }
    assert histories["constant"] == histories["identity"]
    with (
        np.load(tmp_path / "constant/weights.npz") as a,
        np.load(tmp_path / "identity/weights.npz") as b,
    ):
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    assert [
        float(row["next_epoch_learning_rate"]) for row in histories["decay"][:-1]
    ] == pytest.approx([0.01, 0.0055, 0.001])
    assert histories["constant"][1]["nll"] == histories["decay"][1]["nll"]
    assert histories["constant"][2]["nll"] != histories["decay"][2]["nll"]
