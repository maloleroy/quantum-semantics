"""One fit, validation selection, NPZ checkpoints and a separate locked test step."""

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast

import numpy as np
import torch

from .data import Dataset
from .encoding import ContextMethod
from .metrics import Metrics, evaluate
from .models import ModelConfig, ModelKind, WordModel, build_model
from .storage import (
    fingerprint,
    load_weights,
    new_directory,
    read_json,
    save_arrays,
    save_weights,
    write_csv,
    write_json,
)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 0.001
    selection: Literal["success", "nll"] = "success"
    loss: Literal["nll", "success"] = "nll"

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size, self.learning_rate) <= 0:
            raise ValueError("Training budgets and learning rate must be positive")


class FitRecord(TypedDict):
    name: str
    model: dict[str, object]
    seed: int
    selected_epoch: int
    validation: dict[str, object]
    weights_sha256: str
    trainable_parameters: int
    protocol: str


class ModelRecord(TypedDict):
    kind: ModelKind
    size: int
    window: int
    layers: int
    encoder: Literal["auto", "amplitude", "qcse"]
    context_method: ContextMethod
    decoder: Literal["born", "rotated", "marginal"]
    latent: int
    embedding: int
    freeze_core: bool
    ancilla_size: int
    rank: int
    recent_window: int
    amplitude_decay: float
    phase_scale: float


class _Optimizer(Protocol):
    def zero_grad(self) -> None: ...
    def step(self) -> None: ...


def model_config(value: object) -> ModelConfig:
    return ModelConfig(**cast(ModelRecord, value))


def selection_key(metrics: Metrics, criterion: str) -> tuple[float, float]:
    return (
        (metrics.success, -metrics.nll)
        if criterion == "success"
        else (-metrics.nll, metrics.success)
    )


def fit(
    dataset: Dataset, config: ModelConfig, training: TrainConfig, seed: int, output: Path
) -> FitRecord:
    started = time.monotonic()
    model = build_model(config, dataset, seed)
    output.mkdir()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = (
        cast(_Optimizer, torch.optim.Adam(parameters, lr=training.learning_rate))
        if parameters
        else None
    )
    rng = np.random.default_rng(seed)
    best: dict[str, torch.Tensor] = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    best_key, best_epoch = (-float("inf"), -float("inf")), 0
    rows: list[dict[str, object]] = []
    validation = Metrics(0, 0, float("inf"), None, None, None, 0, 0)
    for epoch in range((training.epochs if optimizer is not None else 0) + 1):
        measured, _ = evaluate(model, dataset.validation)
        rows.append({"epoch": epoch, **measured.row()})
        if selection_key(measured, training.selection) > best_key:
            best_key, best_epoch, validation = (
                selection_key(measured, training.selection),
                epoch,
                measured,
            )
            best = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch == training.epochs or optimizer is None:
            break
        order = rng.permutation(len(dataset.train.targets))
        model.train()
        total = 0.0
        for start in range(0, len(order), training.batch_size):
            ids = order[start : start + training.batch_size]
            optimizer.zero_grad()
            probabilities = model(torch.tensor(dataset.train.contexts[ids]))
            targets = torch.tensor(dataset.train.targets[ids])
            correct = probabilities[torch.arange(len(ids)), targets]
            loss = (
                -correct.clamp_min(1e-30).log().mean()
                if training.loss == "nll"
                else -correct.mean()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        rows[-1]["next_epoch_online_loss"] = total / len(order)
    model.load_state_dict(best)
    save_weights(output / "weights.npz", model)
    write_csv(output / "history.csv", rows)
    _, arrays = evaluate(model, dataset.validation)
    save_arrays(output / "validation.npz", arrays)
    train_metrics, _ = evaluate(model, dataset.train)
    record: FitRecord = {
        "name": output.name,
        "model": cast(dict[str, object], asdict(config)),
        "seed": seed,
        "selected_epoch": best_epoch,
        "validation": validation.row(),
        "weights_sha256": fingerprint(output / "weights.npz"),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "protocol": model.protocol,
    }
    write_json(
        output / "fit.json",
        {
            **record,
            "training": asdict(training),
            "selected_training": train_metrics.row(),
            "seconds": time.monotonic() - started,
        },
    )
    print(
        f"{output.name}: validation sample success {validation.success:.2%}; "
        f"argmax {validation.accuracy:.2%}; NLL {validation.nll:.4f}; epoch {best_epoch}",
        flush=True,
    )
    return record


def train(
    dataset_path: Path,
    models: list[ModelConfig],
    training: TrainConfig,
    seeds: list[int],
    root: Path,
) -> Path:
    if not models or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Need models and distinct seeds")
    if len({model.kind for model in models}) != len(models):
        raise ValueError("Use a separate run for duplicate model kinds/configurations")
    dataset = Dataset.load(dataset_path)
    output = new_directory(root, "run")
    dataset.save(output / "dataset.npz")
    source = Path(__file__).parent
    hashes = {path.name: fingerprint(path) for path in source.glob("*.py")}
    settings: dict[str, object] = {
        "dataset_sha256": fingerprint(output / "dataset.npz"),
        "source_sha256": hashes,
        "training": asdict(training),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    write_json(output / "run.json", settings)
    records = [
        fit(dataset, config, training, seed, output / f"{config.kind}-s{seed}")
        for config in models
        for seed in seeds
    ]
    write_json(output / "selection.json", {**settings, "fits": records})
    print(f"Selection locked: {output}. Test has not been scored.", flush=True)
    return output


def selected_models(run: Path) -> tuple[Dataset, list[FitRecord]]:
    selection = read_json(run / "selection.json")
    if fingerprint(run / "dataset.npz") != selection["dataset_sha256"]:
        raise ValueError("Dataset changed after selection")
    hashes = cast(dict[str, str], selection["source_sha256"])
    for name, expected in hashes.items():
        if fingerprint(Path(__file__).with_name(name)) != expected:
            raise ValueError("Model source changed after selection; use the matching checkout")
    records = cast(list[FitRecord], selection["fits"])
    for record in records:
        if fingerprint(run / record["name"] / "weights.npz") != record["weights_sha256"]:
            raise ValueError("Weights changed after selection")
    return Dataset.load(run / "dataset.npz"), records


def test(run: Path) -> Path:
    if (run / "test.csv").exists():
        raise ValueError("Test already scored; inspect test.csv instead of rescoring")
    dataset, records = selected_models(run)
    rows: list[dict[str, object]] = []
    for record in records:
        config = model_config(record["model"])
        model = build_model(config, dataset, record["seed"])
        initial, initial_arrays = evaluate(model, dataset.test)
        load_weights(run / record["name"] / "weights.npz", model)
        measured, arrays = evaluate(model, dataset.test)
        arrays.update({"initial_" + name: value for name, value in initial_arrays.items()})
        save_arrays(run / record["name"] / "test.npz", arrays)
        rows.append(
            {
                "fit": record["name"],
                "protocol": model.protocol,
                **measured.row(),
                "initial_success": initial.success,
                "initial_accuracy": initial.accuracy,
                "initial_nll": initial.nll,
            }
        )
    write_csv(run / "test.csv", rows)
    write_json(
        run / "test.json",
        {
            "selection_sha256": fingerprint(run / "selection.json"),
            "fits": len(records),
            "count": len(dataset.test.targets),
        },
    )
    return run / "test.csv"


def load_predictor(run: Path, name: str) -> tuple[Dataset, WordModel]:
    dataset, records = selected_models(run)
    matches = [record for record in records if record["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Unknown fit {name}")
    record = matches[0]
    model = build_model(model_config(record["model"]), dataset, record["seed"])
    load_weights(run / name / "weights.npz", model)
    return dataset, model
