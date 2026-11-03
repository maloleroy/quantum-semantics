"""Cross-device physics, gradients and portable locked fits; skipped on CPU laptops."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import Dataset, prepare
from src.generation import generate
from src.metrics import evaluate
from src.models import MODEL_KINDS, ModelConfig, ModelKind, build_model
from src.storage import load_weights, read_json, save_weights
from src.training import TrainConfig, load_predictor, train
from src.training import test as score_test

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("QCSE_REQUIRE_CUDA") != "1",
    reason="CUDA checks run on the DGX, not the CPU laptop",
)


@pytest.fixture
def cuda_data(tmp_path: Path) -> Dataset:
    source = Path(__file__).resolve().parents[1] / "data" / "phrases.csv"
    return prepare(source, tmp_path, vocabulary=8, train_limit=24, validation_limit=8, test_limit=8)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_cuda_probabilities_gradients_and_checkpoint(
    kind: ModelKind, cuda_data: Dataset, tmp_path: Path
) -> None:
    torch.set_num_threads(1)
    config = ModelConfig(kind, cuda_data.size, cuda_data.window, rank=4, latent=4)
    cpu = build_model(config, cuda_data, 11)
    gpu = build_model(config, cuda_data, 11, "cuda")
    contexts = torch.tensor(cuda_data.train.contexts[:4])
    expected = cpu(contexts)
    actual = gpu(contexts.cuda())
    assert actual.is_cuda
    assert all(value.is_cuda for value in gpu.state_dict().values())
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-10, rtol=1e-9)
    targets = torch.tensor(cuda_data.train.targets[:4])
    for model, probabilities in ((cpu, expected), (gpu, actual)):
        loss = (
            -probabilities[
                torch.arange(len(targets), device=model.device), targets.to(model.device)
            ]
            .clamp_min(1e-30)
            .log()
            .mean()
        )
        if loss.requires_grad:
            loss.backward()
    for left, right in zip(cpu.parameters(), gpu.parameters(), strict=True):
        if left.grad is not None:
            assert right.grad is not None and right.grad.is_cuda
            torch.testing.assert_close(right.grad.cpu(), left.grad, atol=1e-9, rtol=1e-7)
    save_weights(tmp_path / "weights.npz", gpu)
    restored = build_model(config, cuda_data, 22)
    load_weights(tmp_path / "weights.npz", restored)
    torch.testing.assert_close(restored(contexts), expected, atol=1e-10, rtol=1e-9)
    load_weights(tmp_path / "weights.npz", gpu)
    _, gpu_arrays = evaluate(gpu, cuda_data.validation)
    _, cpu_arrays = evaluate(cpu, cuda_data.validation)
    for key in gpu_arrays:
        np.testing.assert_allclose(
            gpu_arrays[key].astype(np.float64),
            cpu_arrays[key].astype(np.float64),
            atol=1e-9,
            rtol=1e-7,
        )


def test_cuda_locked_training_and_generation(cuda_data: Dataset, tmp_path: Path) -> None:
    configs = [ModelConfig("adaptive", cuda_data.size, cuda_data.window, rank=4)]
    training = TrainConfig(epochs=2, batch_size=16, selection="nll")
    runs: dict[str, Path] = {}
    for device in ("cpu", "cuda"):
        runs[device] = train(
            tmp_path / "dataset.npz", configs, training, [11], tmp_path / device, device
        )
        assert not (runs[device] / "test.csv").exists()
    for filename in ("weights.npz", "validation.npz"):
        with (
            np.load(runs["cpu"] / "adaptive-s11" / filename) as cpu,
            np.load(runs["cuda"] / "adaptive-s11" / filename) as gpu,
        ):
            for key in cpu.files:
                np.testing.assert_allclose(cpu[key], gpu[key], atol=1e-8, rtol=1e-6)
    assert read_json(runs["cuda"] / "run.json")["device"] == "cuda"
    assert score_test(runs["cuda"], "cuda").is_file()
    with pytest.raises(ValueError, match="already scored"):
        score_test(runs["cuda"], "cuda")
    dataset, model = load_predictor(runs["cuda"], "adaptive-s11", "cuda")
    result = generate(dataset, model, "the river", steps=4, seed=7)
    assert result == generate(dataset, model, "the river", steps=4, seed=7)
    assert len(result.tokens) == 4
