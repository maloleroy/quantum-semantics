"""Expected sampling success is distinct from argmax accuracy and NLL."""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch
from torch import Tensor

from .data import FloatArray, IntArray, Partition
from .models import WordModel
from .storage import Array


class _PlotAxis(Protocol):
    def plot(self, x: list[int], y: list[float], *, label: str = "") -> object: ...
    def legend(self) -> object: ...
    def set(self, *, xlabel: str, ylabel: str, title: str = "") -> object: ...


class _FigureOutput(Protocol):
    def savefig(self, path: Path, *, dpi: int) -> None: ...


@dataclass(frozen=True)
class Metrics:
    success: float
    accuracy: float
    nll: float
    known_success: float | None
    known_accuracy: float | None
    known_nll: float | None
    unk_rate: float
    count: int

    def row(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def probability_scores(probabilities: Tensor, targets: Tensor) -> dict[str, Array]:
    if probabilities.ndim != 2 or len(probabilities) != len(targets):
        raise ValueError("Probability/target dimensions differ")
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise ValueError("Non-finite or negative probabilities")
    if torch.max((probabilities.sum(1) - 1).abs()) > 1e-10:
        raise ValueError("Probability mass is not one; do not silently renormalize")
    correct = probabilities[torch.arange(len(targets)), targets]
    maxima = probabilities.max(1, keepdim=True).values
    accuracy = (correct == maxima[:, 0]).double() / (probabilities == maxima).sum(1)
    return {
        "success": cast(Array, correct.numpy()),
        "accuracy": cast(Array, accuracy.numpy()),
        "nll": cast(Array, -correct.clamp_min(1e-30).log().numpy()),
        "targets": cast(Array, targets.numpy()),
    }


def summarize(arrays: dict[str, Array]) -> Metrics:
    targets = cast(IntArray, arrays["targets"])
    known = targets != 0
    success, accuracy, nll = (
        cast(FloatArray, arrays[name]) for name in ("success", "accuracy", "nll")
    )
    return Metrics(
        float(success.mean()),
        float(accuracy.mean()),
        float(nll.mean()),
        float(success[known].mean()) if known.any() else None,
        float(accuracy[known].mean()) if known.any() else None,
        float(nll[known].mean()) if known.any() else None,
        float((~known).mean()),
        len(targets),
    )


def evaluate(
    model: WordModel, part: Partition, batch_size: int = 256
) -> tuple[Metrics, dict[str, Array]]:
    model.eval()
    pieces: dict[str, list[Array]] = {
        name: [] for name in ("success", "accuracy", "nll", "targets")
    }
    with torch.no_grad():
        for start in range(0, len(part.targets), batch_size):
            contexts = torch.tensor(part.contexts[start : start + batch_size])
            targets = torch.tensor(part.targets[start : start + batch_size])
            arrays = probability_scores(model(contexts), targets)
            for name in pieces:
                pieces[name].append(arrays[name])
    combined = {name: np.concatenate(values) for name, values in pieces.items()}
    combined["groups"] = part.groups
    return summarize(combined), combined


def group_interval(
    differences: FloatArray, groups: IntArray, repeats: int = 2000, seed: int = 161803
) -> tuple[float, float]:
    if differences.shape != groups.shape or not len(groups) or repeats < 1:
        raise ValueError("Need paired, nonempty differences and sentence groups")
    _, ids = np.unique(groups, return_inverse=True)
    counts, totals = np.bincount(ids), np.bincount(ids, weights=differences)
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        chosen = rng.integers(len(counts), size=len(counts))
        samples[index] = totals[chosen].sum() / counts[chosen].sum()
    bounds = np.quantile(samples, [0.025, 0.975])
    return float(bounds[0]), float(bounds[1])


def plot_history(history: Path) -> Path:
    from matplotlib.figure import Figure

    with history.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("History is empty")
    epochs = [int(row["epoch"]) for row in rows]
    figure = Figure(figsize=(9, 3.5), layout="constrained")
    scores = cast(_PlotAxis, figure.add_subplot(121))
    scores.plot(epochs, [100 * float(row["success"]) for row in rows], label="Sampling success")
    scores.plot(epochs, [100 * float(row["accuracy"]) for row in rows], label="Argmax accuracy")
    scores.set(xlabel="Epoch", ylabel="Validation (%)", title=history.parent.name)
    scores.legend()
    losses = cast(_PlotAxis, figure.add_subplot(122))
    losses.plot(epochs, [float(row["nll"]) for row in rows])
    losses.set(xlabel="Epoch", ylabel="Validation NLL")
    output = history.with_name("learning.png")
    cast(_FigureOutput, figure).savefig(output, dpi=150)
    return output
