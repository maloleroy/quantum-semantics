"""Factor scaling preserves inference while defining an explicit optimization recipe."""

import csv
import math
import os
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
import torch

from src.data import Dataset, prepare
from src.metrics import evaluate
from src.models import AncillaModel, LowRankModel, ModelConfig, ModelKind, build_model
from src.storage import load_weights, read_json
from src.training import TrainConfig, fit, normalize_factors


@pytest.fixture
def data(tmp_path: Path, sentence_source: Path) -> Dataset:
    return prepare(
        sentence_source,
        tmp_path,
        vocabulary=8,
        train_limit=24,
        validation_limit=8,
        test_limit=8,
    )


def require_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        assert os.environ.get("QCSE_REQUIRE_CUDA") != "1"
        pytest.skip("Factor normalization CUDA checks run on the DGX")
    torch.set_num_threads(1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["lowrank", "prior_content"])
def test_normalization_preserves_probabilities_and_gradient_scaling(
    data: Dataset, device: str, kind: ModelKind
) -> None:
    require_device(device)
    model = build_model(
        ModelConfig(kind, data.size, data.window, rank=3, recent_window=2), data, 11, device
    )
    assert isinstance(model, (AncillaModel, LowRankModel))
    with torch.no_grad():
        model.energies.normal_(std=0.3)
        model.factors.mul_(torch.tensor([2.0, 4.0, 7.0], device=device)[None, None, :, None])
    contexts = torch.tensor(data.validation.contexts, device=device)
    targets = torch.tensor(data.validation.targets, device=device)
    ids = torch.arange(len(contexts), device=device)
    before = model(contexts)
    gradient = torch.autograd.grad(-before[ids, targets].log().mean(), model.factors)[0]
    original = model.factors.detach().clone()
    rng = torch.random.get_rng_state().clone()
    normalize_factors(model, None)
    torch.testing.assert_close(original, model.factors, atol=0, rtol=0)
    torch.testing.assert_close(rng, torch.random.get_rng_state(), atol=0, rtol=0)
    scale = math.sqrt(2) / original.square().sum((1, 3)).sqrt()
    normalize_factors(model, math.sqrt(2))
    after = model(contexts)
    actual = torch.autograd.grad(-after[ids, targets].log().mean(), model.factors)[0]
    torch.testing.assert_close(before, after, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual, gradient / scale[:, None, :, None], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(
        model.factors.square().sum((1, 3)), torch.full_like(scale, 2), atol=1e-12, rtol=1e-12
    )


def test_reject_invalid_targets_and_unsupported_models(data: Dataset, tmp_path: Path) -> None:
    for target in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            TrainConfig(factor_norm=target)
    for config in (
        ModelConfig("mlp", data.size, data.window),
        ModelConfig("prior", data.size, data.window, rank=3, freeze_core=True),
    ):
        output = tmp_path / config.kind
        with pytest.raises(ValueError, match="trainable spectral factors"):
            fit(data, config, TrainConfig(factor_norm=math.sqrt(2)), 11, output)
        assert not output.exists()
    model = build_model(ModelConfig("prior", data.size, data.window, rank=3), data, 11)
    assert isinstance(model, AncillaModel)
    with torch.no_grad():
        model.factors[:, :, 0].zero_()
    with pytest.raises(ValueError, match="zero or nonfinite"):
        normalize_factors(model, math.sqrt(2))


class Stepper(Protocol):
    def zero_grad(self) -> None: ...
    def step(self) -> None: ...


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("target", [None, math.sqrt(2)])
def test_fit_matches_independent_adam_with_unmodified_moments(
    data: Dataset, tmp_path: Path, device: str, target: float | None
) -> None:
    require_device(device)
    config = ModelConfig("prior_content", data.size, data.window, rank=3, recent_window=2)
    settings = TrainConfig(
        epochs=2,
        batch_size=3,
        learning_rate=0.03,
        selection="nll",
        epoch_examples=5,
        factor_norm=target,
    )
    model = build_model(config, data, 11, device)
    assert isinstance(model, AncillaModel)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if target is not None:
        with torch.no_grad():
            model.factors.mul_(
                target / model.factors.square().sum(3, keepdim=True).sum(1, keepdim=True).sqrt()
            )
    optimizer = cast(Stepper, torch.optim.Adam(parameters, lr=0.03))
    rng = np.random.default_rng(11)
    expected: list[float] = []
    best = (float("inf"), float("inf"))
    selected: dict[str, torch.Tensor] = {}
    selected_epoch = -1
    for epoch in range(3):
        metrics, _ = evaluate(model, data.validation, 3)
        expected.append(metrics.nll)
        key = (metrics.nll, -metrics.success)
        if key < best:
            best = key
            selected_epoch = epoch
            selected = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch == 2:
            break
        order = rng.permutation(len(data.train.targets))[:5]
        for ids in (order[:3], order[3:]):
            optimizer.zero_grad()
            probabilities = model(torch.tensor(data.train.contexts[ids], device=device))
            labels = torch.tensor(data.train.targets[ids], device=device)
            loss = (
                -probabilities[torch.arange(len(ids), device=device), labels]
                .clamp_min(1e-30)
                .log()
                .mean()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5)
            optimizer.step()
            if target is not None:
                with torch.no_grad():
                    model.factors.mul_(
                        target
                        / model.factors.square().sum(3, keepdim=True).sum(1, keepdim=True).sqrt()
                    )
    output = tmp_path / "fit"
    record = fit(data, config, settings, 11, output, device)
    with (output / "history.csv").open() as file:
        history = list(csv.DictReader(file))
    np.testing.assert_allclose(
        [float(row["nll"]) for row in history], expected, atol=1e-12, rtol=1e-12
    )
    assert record["selected_epoch"] == selected_epoch
    for row in history:
        if target is None:
            assert "factor_norm_mean" not in row
        else:
            for key in ("factor_norm_min", "factor_norm_mean", "factor_norm_max"):
                assert float(row[key]) == pytest.approx(target, abs=1e-12)
    load_weights(output / "weights.npz", model)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, selected[name], atol=1e-12, rtol=1e-12)
    metadata = read_json(output / "fit.json")
    training = cast(dict[str, object], metadata["training"])
    assert training["factor_norm"] == target
    assert not (output / "test.npz").exists()
