"""Small context-conditioned Hamiltonian residual; validation-only CPU proof of concept.

Run with --toy, or --dataset PATH to a prepared Hamiltonian-study dataset.
This latent-state model is separate from the fixed vocabulary-graph walk.
"""

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qcse.outputs import new_run_directory, save_npz, write_json

from .data import transition_probabilities
from .run import metrics

KINDS = ("hamiltonian", "classical", "last_word")


@dataclass(frozen=True)
class ResidualConfig:
    vocabulary: int
    window: int
    embedding: int = 16
    hidden: int = 32
    features: int = 8
    unknown_id: int | None = 0


@dataclass
class ContextData:
    train_contexts: torch.Tensor
    train_targets: torch.Tensor
    train_groups: np.ndarray
    validation_contexts: torch.Tensor
    validation_targets: torch.Tensor
    validation_groups: np.ndarray
    prior: torch.Tensor
    vocabulary: list[str]
    provenance: dict


def directional_generators(dimension: int) -> torch.Tensor:
    """Fixed Hermitian source/target couplings in a latent register, norm <= 1.

    Latent basis states are not vocabulary IDs. No corpus graph or word meaning
    is attributed to these fixed generators; the learned context controls them.
    """
    if dimension < 4 or dimension % 2:
        raise ValueError("The latent dimension must be even and at least four")
    rng = np.random.default_rng(723)
    half = dimension // 2
    generators: list[np.ndarray] = [np.diag(np.r_[-np.ones(half), np.ones(half)])]
    for _ in range(dimension - 1):
        coupling = rng.normal(size=(half, half)) + 1j * rng.normal(size=(half, half))
        matrix = np.zeros((dimension, dimension), dtype=complex)
        matrix[:half, half:] = coupling
        matrix[half:, :half] = coupling.conj().T
        generators.append(matrix / np.max(np.abs(np.linalg.eigvalsh(matrix))))
    return torch.tensor(np.asarray(generators), dtype=torch.complex64)


def hamiltonian_features(coefficients: torch.Tensor, generators: torch.Tensor) -> torch.Tensor:
    """Full Born distribution after exp(-i H(context)); no postselection."""
    hamiltonian = torch.einsum("bk,kij->bij", coefficients.to(generators.dtype), generators)
    unitary = torch.matrix_exp(-1j * hamiltonian)
    initial = torch.ones(generators.shape[-1], dtype=generators.dtype, device=generators.device)
    initial = initial / math.sqrt(len(initial))
    state = unitary @ initial
    return state.abs().square()


class ContextResidual(nn.Module):
    """Identical learned context encoder/head across Hamiltonian/classical controls."""

    prior: torch.Tensor
    generators: torch.Tensor

    def __init__(self, config: ResidualConfig, prior: torch.Tensor, kind: str):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(kind)
        self.config, self.kind = config, kind
        self.lookup = nn.Embedding(config.vocabulary + 1, config.embedding, config.vocabulary)
        self.encoder = nn.Sequential(
            nn.Linear(config.window * config.embedding, config.hidden),
            nn.Tanh(),
            nn.Linear(config.hidden, config.features),
        )
        self.head = nn.Linear(config.features, config.vocabulary)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("prior", prior.detach().clone())
        self.register_buffer("generators", directional_generators(config.features))

    def features(self, contexts: torch.Tensor) -> torch.Tensor:
        if self.kind == "last_word":
            contexts = torch.cat((torch.full_like(contexts[:, :-1], -1), contexts[:, -1:]), 1)
        ids = torch.where(contexts >= 0, contexts, self.config.vocabulary)
        encoded = self.lookup(ids).flatten(1)
        coefficients = torch.tanh(self.encoder(encoded))
        if self.kind == "hamiltonian":
            return hamiltonian_features(coefficients, self.generators)
        return coefficients.softmax(-1)

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        return self.prior[contexts[:, -1]].log() + self.head(self.features(contexts))


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def load_data(directory: Path, train_limit: int, validation_limit: int) -> ContextData:
    path = directory / "dataset.npz"
    provenance = json.loads((directory / "dataset.json").read_text())
    arrays = np.load(path)
    # Deliberately never read the test arrays. Existing training/validation groups
    # and training-only vocabulary/statistics are inherited without recomputation.
    unigram, transitions = arrays["unigram"], arrays["transitions"]
    selected: dict[str, tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}
    for split, limit in (("train", train_limit), ("validation", validation_limit)):
        contexts, targets, groups = (
            arrays[f"{split}_contexts"],
            arrays[f"{split}_targets"],
            arrays[f"{split}_groups"],
        )
        ids = np.arange(len(targets))
        if limit and limit < len(ids):
            ids = np.sort(np.random.default_rng(1618).choice(ids, limit, replace=False))
        selected[split] = (torch.tensor(contexts[ids]), torch.tensor(targets[ids]), groups[ids])
    assert not np.intersect1d(selected["train"][2], selected["validation"][2]).size
    prior = torch.tensor(transition_probabilities(transitions, unigram), dtype=torch.float32)
    return ContextData(
        train_contexts=selected["train"][0],
        train_targets=selected["train"][1],
        train_groups=selected["train"][2],
        validation_contexts=selected["validation"][0],
        validation_targets=selected["validation"][1],
        validation_groups=selected["validation"][2],
        prior=prior,
        vocabulary=arrays["vocabulary"].tolist(),
        provenance={
            "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": provenance,
            "train_eligible_targets": len(arrays["train_targets"]),
            "train_limit": train_limit,
            "validation_limit": validation_limit,
            "bigram_strength": 10.0,
        },
    )


def toy_data() -> ContextData:
    """The first cue determines the next word; the last word is uninformative.

    All 256 contexts are unique. Split complete context/target sentence groups;
    validation uses unseen combinations of familiar cues and filler words.
    """
    vocabulary = (
        [f"cue{i}" for i in range(4)]
        + [f"answer{i}" for i in range(4)]
        + [f"filler{i}" for i in range(16)]
        + ["bridge"]
    )
    contexts = np.array(
        [[a, b, filler, 24] for a in range(4) for b in range(4) for filler in range(8, 24)]
    )
    targets = contexts[:, 0] + 4
    order = np.random.default_rng(31415).permutation(len(targets))
    train, validation = order[:192], order[192:]
    transitions = np.zeros((len(vocabulary), len(vocabulary)))
    counts = np.zeros(len(vocabulary))
    for index in train:
        sequence = np.r_[contexts[index], targets[index]]
        np.add.at(transitions, (sequence[:-1], sequence[1:]), 1)
        np.add.at(counts, sequence[1:], 1)
    unigram = (counts + 0.5) / (counts.sum() + 0.5 * len(counts))
    prior = torch.tensor(transition_probabilities(transitions, unigram), dtype=torch.float32)
    return ContextData(
        torch.tensor(contexts[train]),
        torch.tensor(targets[train]),
        train,
        torch.tensor(contexts[validation]),
        torch.tensor(targets[validation]),
        validation,
        prior,
        vocabulary,
        {"task": "first cue predicts answer; shared final bridge", "split_seed": 31415},
    )


def score(model: ContextResidual, contexts: torch.Tensor, targets: torch.Tensor) -> dict:
    with torch.no_grad():
        probabilities = torch.cat([model(chunk).softmax(-1) for chunk in contexts.split(256)])
    result, _ = metrics(probabilities.numpy(), targets.numpy())
    # Existing metrics call class 0 UNK. The toy has no UNK and must not use that convention.
    if model.config.unknown_id is None:
        result.pop("known_nll")
        result.pop("known_top1")
        result.pop("unk_rate")
    return result


def fit(
    data: ContextData,
    config: ResidualConfig,
    kind: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    output: Path,
    weight_decay: float = 0.001,
) -> dict:
    started = time.monotonic()
    torch.manual_seed(seed)
    model = ContextResidual(config, data.prior, kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    initial = score(model, data.validation_contexts, data.validation_targets)
    history = [{"epoch": 0, "validation": initial}]
    best = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_nll, best_epoch = initial["nll"], 0
    gradients = {}
    parameter_hashes = {name: tensor_hash(value) for name, value in model.named_parameters()}
    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(len(data.train_targets))
        total_loss = 0.0
        for start in range(0, len(permutation), batch_size):
            ids = permutation[start : start + batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data.train_contexts[ids]), data.train_targets[ids])
            loss.backward()
            if epoch == 2 and start == 0:
                gradients = {
                    name: float(value.grad.norm())
                    for name, value in model.named_parameters()
                    if value.grad is not None
                }
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
        validation = score(model, data.validation_contexts, data.validation_targets)
        history.append(
            {
                "epoch": epoch,
                "train_online_nll": total_loss / len(permutation),
                "validation": validation,
            }
        )
        if validation["nll"] < best_nll:
            best_nll, best_epoch = validation["nll"], epoch
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best)
    selected = score(model, data.validation_contexts, data.validation_targets)
    training = score(model, data.train_contexts, data.train_targets)
    result = {
        "kind": kind,
        "seed": seed,
        "optimizer": {"lr": lr, "weight_decay": weight_decay},
        "epochs": epochs,
        "batch_size": batch_size,
        "parameters": sum(value.numel() for value in model.parameters()),
        "initial_parameter_hashes": parameter_hashes,
        "selected_epoch": best_epoch,
        "initial_validation": initial,
        "selected_validation": selected,
        "selected_training": training,
        "epoch2_gradient_norms": gradients,
        "history": history,
        "seconds": time.monotonic() - started,
    }
    name = f"{kind}-s{seed}"
    torch.save({"config": asdict(config), "kind": kind, "state": best}, output / f"{name}.pt")
    write_json(output / f"{name}.json", result)
    print(
        f"{name}: validation NLL {initial['nll']:.5f} -> {selected['nll']:.5f}, "
        f"top-1 {100 * selected['top1']:.2f}%, epoch {best_epoch}, {result['seconds']:.1f}s",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--toy", action="store_true")
    source.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/contextual-hamiltonian"))
    parser.add_argument("--train-limit", type=int, default=32768)
    parser.add_argument("--validation-limit", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    args = parser.parse_args()
    if (
        min(args.epochs, args.batch_size, args.lr) <= 0
        or min(args.train_limit, args.validation_limit, args.weight_decay) < 0
    ):
        parser.error("Epochs, batch size and LR must be positive; limits/decay nonnegative")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Training seeds must be distinct")
    torch.set_num_threads(1)
    data = (
        toy_data() if args.toy else load_data(args.dataset, args.train_limit, args.validation_limit)
    )
    config = ResidualConfig(
        len(data.vocabulary), data.train_contexts.shape[1], unknown_id=None if args.toy else 0
    )
    runtime = {"device": "cpu", "torch": torch.__version__, "protocol": "validation-only POC"}
    with new_run_directory(args.output, "toy" if args.toy else "corpus", runtime) as output:
        settings = vars(args) | {"output": str(args.output), "dataset": str(args.dataset)}
        write_json(
            output / "protocol.json",
            {
                "settings": settings,
                "model": asdict(config),
                "data": data.provenance,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "vocabulary": data.vocabulary,
                "test_scored": False,
                "selection": (
                    "Minimum full validation NLL, including epoch zero; fixed shared recipe"
                ),
            },
        )
        save_npz(
            output / "selection.npz",
            train_contexts=data.train_contexts.numpy(),
            train_targets=data.train_targets.numpy(),
            train_groups=data.train_groups,
            validation_contexts=data.validation_contexts.numpy(),
            validation_targets=data.validation_targets.numpy(),
            validation_groups=data.validation_groups,
        )
        records = []
        for seed in args.seeds:
            for kind in KINDS:
                records.append(
                    fit(
                        data,
                        config,
                        kind,
                        seed,
                        args.epochs,
                        args.batch_size,
                        args.lr,
                        output,
                        args.weight_decay,
                    )
                )
        paired = [row["initial_parameter_hashes"] for row in records]
        for index in range(0, len(paired), len(KINDS)):
            assert all(value == paired[index] for value in paired[index : index + len(KINDS)])
        write_json(
            output / "summary.json",
            {"protocol": "validation-only; no test scoring", "records": records},
        )


if __name__ == "__main__":
    main()
