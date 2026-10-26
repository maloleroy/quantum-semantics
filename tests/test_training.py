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
