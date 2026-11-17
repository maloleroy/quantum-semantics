"""Cross-device physics, gradients and portable locked fits; skipped on CPU laptops."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import Dataset, prepare
from src.generation import generate
from src.metrics import evaluate, probability_scores
from src.models import MODEL_KINDS, ModelConfig, ModelKind, build_model
from src.storage import load_weights, read_json, save_weights
from src.training import TrainConfig, load_predictor, train
from src.training import test as score_test

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("QCSE_REQUIRE_CUDA") != "1",
    reason="CUDA checks run on the DGX, not the CPU laptop",
)


@pytest.fixture
def cuda_data(tmp_path: Path, sentence_source: Path) -> Dataset:
    return prepare(
        sentence_source, tmp_path, vocabulary=8, train_limit=24, validation_limit=8, test_limit=8
    )


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_cuda_probabilities_gradients_and_checkpoint(
    kind: ModelKind, cuda_data: Dataset, tmp_path: Path
) -> None:
    torch.set_num_threads(1)
    config = ModelConfig(kind, cuda_data.size, cuda_data.window, rank=4, latent=4)
    cpu = build_model(config, cuda_data, 11)
    gpu = build_model(config, cuda_data, 11, "cuda")
    if kind.startswith("prior"):
        # Exercise learned dynamics rather than identity priors with exact ties.
        # Argmax tie scores are discontinuous under CPU/CUDA roundoff.
        with torch.no_grad():
            for name, parameter in cpu.named_parameters():
                if name in {"energies", "spectral_angles", "key_angles", "correction.weight"}:
                    parameter.normal_(std=0.1)
        gpu.load_state_dict(cpu.state_dict())
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
    if kind.startswith("prior"):
        with torch.no_grad():
            validation = cpu(torch.tensor(cuda_data.validation.contexts))
        top = validation.topk(2, dim=1).values
        assert torch.all(top[:, 0] - top[:, 1] > 1e-10)
    for key in gpu_arrays:
        np.testing.assert_allclose(
            gpu_arrays[key].astype(np.float64),
            cpu_arrays[key].astype(np.float64),
            atol=1e-9,
            rtol=1e-7,
        )


def test_cuda_exact_tie_scores() -> None:
    probabilities = torch.tensor([[0.5, 0.5, 0.0], [0.25, 0.25, 0.5]], dtype=torch.float64)
    targets = torch.tensor([0, 1])
    cpu = probability_scores(probabilities, targets)
    gpu = probability_scores(probabilities.cuda(), targets.cuda())
    np.testing.assert_array_equal(cpu["accuracy"], [0.5, 0.0])
    for key in cpu:
        np.testing.assert_allclose(
            gpu[key].astype(np.float64), cpu[key].astype(np.float64), atol=1e-15, rtol=1e-15
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
