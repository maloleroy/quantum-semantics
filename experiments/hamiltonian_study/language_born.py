"""Fixed ordered word amplitudes, trainable Hamiltonian, direct Born word readout."""

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from qcse.outputs import new_run_directory, save_npz, write_json

from .born_evidence import configure_worker, digest, hermitian
from .data import prepare, transition_probabilities
from .run import metrics

KINDS = ("coherent", "dephased", "mlp", "calibrated_bigram")


class HermitianEvolution(torch.autograd.Function):
    """Exact spectral exp(-iH), with a gap-safe first-order Frechet adjoint.

    The divided difference is -i exp(-i(a+b)/2) sinc((a-b)/(2 pi)).
    It has the correct continuous limit at repeated eigenvalues. Eigenvectors
    themselves are never differentiated. Higher-order gradients are not supported.
    """

    @staticmethod
    def forward(ctx, matrix):
        eigenvalues, vectors = torch.linalg.eigh(matrix)
        ctx.save_for_backward(eigenvalues, vectors)
        return (vectors * torch.exp(-1j * eigenvalues)) @ vectors.mH

    @staticmethod
    def backward(ctx, gradient):
        eigenvalues, vectors = ctx.saved_tensors
        difference = eigenvalues[:, None] - eigenvalues[None, :]
        midpoint = (eigenvalues[:, None] + eigenvalues[None, :]) / 2
        divided = -1j * torch.exp(-1j * midpoint) * torch.sinc(difference / (2 * torch.pi))
        adjoint = vectors @ (divided.conj() * (vectors.mH @ gradient @ vectors)) @ vectors.mH
        return (adjoint + adjoint.mH) / 2


def encode(contexts: torch.Tensor, vocabulary: int) -> torch.Tensor:
    """Fixed recency and position phases; combine repeated words, then normalize."""
    if contexts.ndim != 2 or torch.any(contexts >= vocabulary) or torch.any(contexts < -1):
        raise ValueError("Contexts must be word IDs or -1 padding")
    ages = torch.arange(contexts.shape[1] - 1, -1, -1, dtype=torch.float64)
    coefficients = (0.7**ages).sqrt() * torch.exp(0.5j * torch.pi * ages)
    amplitudes = torch.zeros((len(contexts), vocabulary), dtype=torch.complex128)
    amplitudes.scatter_add_(1, contexts.clamp_min(0), (contexts >= 0) * coefficients)
    norms = amplitudes.norm(dim=1, keepdim=True)
    if torch.any(norms < 1e-12):
        raise ValueError("Empty/cancelled context has no normalized encoding")
    return amplitudes / norms


class HamiltonianWords(nn.Module):
    def __init__(self, vocabulary: int, dephased: bool = False):
        super().__init__()
        self.vocabulary, self.dephased = vocabulary, dephased
        self.coefficients = nn.Parameter(torch.randn(vocabulary**2, dtype=torch.float64) * 0.03)

    def operator(self) -> torch.Tensor:
        return HermitianEvolution.apply(hermitian(self.coefficients, self.vocabulary))

    def forward(self, states, last_words, operator=None):
        unitary = self.operator() if operator is None else operator
        if self.dephased:
            return states.abs().square() @ unitary.abs().square().T
        return (states @ unitary.T).abs().square()


class ClassicalWords(nn.Module):
    def __init__(self, vocabulary: int):
        super().__init__()
        hidden = (vocabulary**2 - vocabulary) // (3 * vocabulary + 1)
        self.network = nn.Sequential(
            nn.Linear(2 * vocabulary, hidden), nn.Tanh(), nn.Linear(hidden, vocabulary)
        ).double()

    def forward(self, states, last_words, operator=None):
        return self.network(torch.cat((states.real, states.imag), 1)).softmax(-1)


class CalibratedBigram(nn.Module):
    prior: torch.Tensor
    unigram: torch.Tensor

    def __init__(self, prior, unigram):
        super().__init__()
        self.register_buffer("prior", prior.clone())
        self.register_buffer("unigram", unigram.clone())
        self.log_temperature = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.mixture_logit = nn.Parameter(torch.tensor(-6.0, dtype=torch.float64))

    def forward(self, states, last_words, operator=None):
        calibrated = (self.prior[last_words].log() / self.log_temperature.exp()).softmax(-1)
        mixture = self.mixture_logit.sigmoid()
        return (1 - mixture) * calibrated + mixture * self.unigram


def make_model(kind, vocabulary, prior, unigram):
    if kind in {"coherent", "dephased"}:
        return HamiltonianWords(vocabulary, kind == "dephased")
    if kind == "mlp":
        return ClassicalWords(vocabulary)
    if kind == "calibrated_bigram":
        return CalibratedBigram(prior, unigram)
    raise ValueError(kind)


@dataclass
class Partition:
    states: torch.Tensor
    targets: torch.Tensor
    last_words: torch.Tensor
    groups: np.ndarray


def load_partition(dataset: Path, split: str, vocabulary: int) -> Partition:
    with np.load(dataset / "dataset.npz") as data:
        contexts = torch.tensor(data[f"{split}_contexts"])
        targets, groups = torch.tensor(data[f"{split}_targets"]), data[f"{split}_groups"]
    return Partition(encode(contexts, vocabulary), targets, contexts[:, -1], groups)


def load_statistics(dataset: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    with np.load(dataset / "dataset.npz") as data:
        unigram = data["unigram"]
        prior = transition_probabilities(data["transitions"], unigram)
        words = data["vocabulary"].tolist()
    return torch.tensor(prior), torch.tensor(unigram), words


def aggregate(arrays: dict[str, np.ndarray]) -> dict:
    targets, loss = arrays["targets"], arrays["nll"]
    known = targets != 0
    return {
        "nll": float(loss.mean()),
        "perplexity": float(np.exp(loss.mean())),
        "top1": float(arrays["top1"].mean()),
        "top5": float(arrays["top5"].mean()),
        "known_nll": float(loss[known].mean()) if known.any() else None,
        "known_top1": float(arrays["top1"][known].mean()) if known.any() else None,
        "unk_rate": float((~known).mean()),
        "count": len(targets),
    }


def score(model, part: Partition) -> tuple[dict, dict[str, np.ndarray]]:
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in ("nll", "top1", "top5", "targets")}
    max_norm_error = 0.0
    with torch.no_grad():
        operator = model.operator() if isinstance(model, HamiltonianWords) else None
        for start in range(0, len(part.targets), 1024):
            end = start + 1024
            probabilities = model(part.states[start:end], part.last_words[start:end], operator)
            if not torch.isfinite(probabilities).all() or torch.min(probabilities) < 0:
                raise ValueError("Invalid probabilities")
            max_norm_error = max(max_norm_error, float((probabilities.sum(1) - 1).abs().max()))
            if max_norm_error > 1e-10:
                raise ValueError("Probability mass changed; no output renormalization is allowed")
            _, arrays = metrics(probabilities.numpy(), part.targets[start:end].numpy())
            for key in pieces:
                pieces[key].append(arrays[key])
    joined = {key: np.concatenate(value) for key, value in pieces.items()}
    return aggregate(joined) | {"max_norm_error": max_norm_error}, joined


@dataclass(frozen=True)
class Job:
    dataset: str
    output: str
    kind: str
    seed: int
    lr: float
    epochs: int
    batch_size: int


def fit(job: Job) -> dict:
    torch.manual_seed(job.seed)
    started = time.monotonic()
    dataset, output = Path(job.dataset), Path(job.output)
    prior, unigram, words = load_statistics(dataset)
    training = load_partition(dataset, "train", len(words))
    validation = load_partition(dataset, "validation", len(words))
    if set(training.groups) & set(validation.groups):
        raise ValueError("Training and validation groups overlap")
    model = make_model(job.kind, len(words), prior, unigram)
    output.mkdir()
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    hashes = {
        key: hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
        for key, value in model.named_parameters()
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=job.lr)
    rng = np.random.default_rng(job.seed)
    best_nll, best_epoch, best = float("inf"), 0, initial
    history = []
    first_gradient = None
    for epoch in range(job.epochs + 1):
        measured, _ = score(model, validation)
        history.append({"epoch": epoch, "validation": measured})
        if measured["nll"] < best_nll:
            best_nll, best_epoch = measured["nll"], epoch
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
        write_json(
            output / "progress.json",
            {"epoch": epoch, "best_epoch": best_epoch, "best_validation_nll": best_nll},
        )
        if epoch == job.epochs:
            break
        permutation = rng.permutation(len(training.targets))
        total_loss = 0.0
        for start in range(0, len(permutation), job.batch_size):
            ids = permutation[start : start + job.batch_size]
            optimizer.zero_grad()
            probabilities = model(training.states[ids], training.last_words[ids])
            loss = (
                -probabilities[torch.arange(len(ids)), training.targets[ids]]
                .clamp_min(1e-30)
                .log()
                .mean()
            )
            loss.backward()
            if first_gradient is None:
                first_gradient = (
                    sum(
                        float(p.grad.square().sum())
                        for p in model.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
        history[-1]["next_epoch_online_train_nll"] = total_loss / len(permutation)
    model.load_state_dict(best)
    selected_validation, _ = score(model, validation)
    selected_training, _ = score(model, training)
    torch.save(
        {"kind": job.kind, "vocabulary": len(words), "state": best, "initial": initial},
        output / "weights.pt",
    )
    result = {
        "job": asdict(job),
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_parameter_hashes": hashes,
        "first_gradient_norm": first_gradient,
        "selected_epoch": best_epoch,
        "initial_validation": history[0]["validation"],
        "selected_validation": selected_validation,
        "selected_training": selected_training,
        "history": history,
        "weights_sha256": digest(output / "weights.pt"),
        "seconds": time.monotonic() - started,
    }
    write_json(output / "result.json", result)
    return result


def execute(jobs: list[Job], workers: int) -> list[dict]:
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=configure_worker,
    ) as pool:
        futures = [pool.submit(fit, job) for job in jobs]
        records = []
        for future in as_completed(futures):
            result = future.result()
            records.append(result)
            job, values = result["job"], result["selected_validation"]
            print(
                f"{job['kind']} s{job['seed']} lr={job['lr']}: val NLL {values['nll']:.5f}, "
                f"accuracy {values['top1']:.2%}, epoch {result['selected_epoch']}",
                flush=True,
            )
    return sorted(
        records, key=lambda row: (row["job"]["kind"], row["job"]["seed"], row["job"]["lr"])
    )


def group_interval(values: np.ndarray, groups: np.ndarray) -> list[float]:
    _, ids = np.unique(groups, return_inverse=True)
    counts = np.bincount(ids)
    totals = np.bincount(ids, weights=values)
    rng = np.random.default_rng(161803)
    estimates = []
    for _ in range(2000):
        chosen = rng.integers(len(counts), size=len(counts))
        estimates.append(totals[chosen].sum() / counts[chosen].sum())
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def evaluate(output: Path) -> dict:
    lock = json.loads((output / "selection_lock.json").read_text())
    if (output / "test_results.json").exists():
        raise ValueError("Test already scored; use the saved results")
    for relative, expected in lock["source_sha256"].items():
        if digest(Path(relative)) != expected:
            raise ValueError("Source changed after selection lock")
    if digest(output / "dataset.npz") != lock["dataset_sha256"]:
        raise ValueError("Dataset changed after selection lock")
    prior, unigram, words = load_statistics(output)
    testing = load_partition(output, "test", len(words))
    with np.load(output / "dataset.npz") as data:
        if set(testing.groups) & (set(data["train_groups"]) | set(data["validation_groups"])):
            raise ValueError("Test groups overlap fit groups")
    records, predictions = [], {}
    for row in lock["fits"]:
        job = row["job"]
        path = Path(job["output"]) / "weights.pt"
        if digest(path) != row["weights_sha256"]:
            raise ValueError("Checkpoint changed after selection lock")
        checkpoint = torch.load(path, weights_only=True)
        model = make_model(job["kind"], len(words), prior, unigram)
        model.load_state_dict(checkpoint["state"])
        selected, arrays = score(model, testing)
        model.load_state_dict(checkpoint["initial"])
        initial, initial_arrays = score(model, testing)
        key = f"{job['kind']}-s{job['seed']}"
        predictions.update({f"{key}-{name}": values for name, values in arrays.items()})
        predictions.update(
            {f"initial-{key}-{name}": values for name, values in initial_arrays.items()}
        )
        records.append(
            {
                "kind": job["kind"],
                "seed": job["seed"],
                "parameters": row["parameters"],
                "selected_epoch": row["selected_epoch"],
                "test": selected,
                "initial_test": initial,
            }
        )
    with np.load(output / "dataset.npz") as data:
        sampled_counts = np.zeros((len(words), len(words)))
        np.add.at(sampled_counts, (data["train_contexts"][:, -1], data["train_targets"]), 1)
        target_counts = np.bincount(data["train_targets"], minlength=len(words))
        sampled_unigram = (target_counts + 0.5) / (target_counts.sum() + 0.5 * len(words))
        sampled_prior = torch.tensor(transition_probabilities(sampled_counts, sampled_unigram))
    baselines = {}
    with torch.no_grad():
        for kind in ("identity", "unigram", "bigram", "subset_bigram", "directed_pool"):
            pieces: dict[str, list[np.ndarray]] = {
                key: [] for key in ("nll", "top1", "top5", "targets")
            }
            for start in range(0, len(testing.targets), 1024):
                end = start + 1024
                states, last = testing.states[start:end], testing.last_words[start:end]
                if kind == "identity":
                    probabilities = states.abs().square()
                elif kind == "unigram":
                    probabilities = unigram.expand(len(states), -1)
                elif kind == "bigram":
                    probabilities = prior[last]
                elif kind == "subset_bigram":
                    probabilities = sampled_prior[last]
                else:
                    probabilities = states.abs().square() @ prior
                _, arrays = metrics(probabilities.numpy(), testing.targets[start:end].numpy())
                for name, values in arrays.items():
                    pieces[name].append(values)
            joined = {name: np.concatenate(values) for name, values in pieces.items()}
            baselines[kind] = aggregate(joined)
            predictions.update({f"{kind}-{name}": values for name, values in joined.items()})
    summary = {}
    for kind in KINDS:
        selected = [row for row in records if row["kind"] == kind]
        summary[kind] = {"parameters": selected[0]["parameters"]}
        for key in ("nll", "top1", "top5", "known_nll", "known_top1"):
            values = [row["test"][key] for row in selected]
            summary[kind][key + "_mean"] = float(np.mean(values))
            summary[kind][key + "_sd"] = float(np.std(values, ddof=1))
            summary[kind]["initial_" + key + "_mean"] = float(
                np.mean([row["initial_test"][key] for row in selected])
            )
    intervals = {}
    coherent = np.mean([predictions[f"coherent-s{seed}-nll"] for seed in lock["seeds"]], axis=0)
    for kind in ("dephased", "mlp", "calibrated_bigram"):
        other = np.mean([predictions[f"{kind}-s{seed}-nll"] for seed in lock["seeds"]], axis=0)
        delta = coherent - other
        intervals[kind] = {
            "coherent_minus_other_nll": float(delta.mean()),
            "group_bootstrap_95": group_interval(delta, testing.groups),
        }
    frozen = np.mean(
        [predictions[f"initial-coherent-s{seed}-nll"] for seed in lock["seeds"]], axis=0
    )
    delta = coherent - frozen
    known = testing.targets.numpy() != 0
    intervals["frozen"] = {
        "coherent_trained_minus_initial_nll": float(delta.mean()),
        "group_bootstrap_95": group_interval(delta, testing.groups),
        "known_delta_nll": float(delta[known].mean()),
        "known_group_bootstrap_95": group_interval(delta[known], testing.groups[known]),
    }
    save_npz(output / "test_arrays.npz", **predictions, groups=testing.groups)
    result = {
        "selection_lock_sha256": digest(output / "selection_lock.json"),
        "fits": records,
        "summary": summary,
        "baselines": baselines,
        "intervals": intervals,
        "test_unk_rate": float((testing.targets == 0).double().mean()),
        "test_sentence_groups": len(np.unique(testing.groups)),
    }
    write_json(output / "test_results.json", result)
    return result


def analyze(output: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lock = json.loads((output / "selection_lock.json").read_text())
    result = json.loads((output / "test_results.json").read_text())
    assert result["selection_lock_sha256"] == digest(output / "selection_lock.json")
    assert lock["dataset_sha256"] == digest(output / "dataset.npz")
    assert lock["recipe_lock_sha256"] == digest(output / "recipe_lock.json")
    for relative, expected in lock["source_sha256"].items():
        assert digest(output / "source" / relative) == expected
    for row in lock["fits"]:
        assert row["selected_validation"]["nll"] == min(
            item["validation"]["nll"] for item in row["history"]
        )
        assert row["weights_sha256"] == digest(Path(row["job"]["output"]) / "weights.pt")
    for seed in lock["seeds"]:
        hashes = [
            row["initial_parameter_hashes"]
            for row in lock["fits"]
            if row["job"]["seed"] == seed and row["job"]["kind"] in {"coherent", "dephased"}
        ]
        assert len(hashes) == 2 and hashes[0] == hashes[1]
    with np.load(output / "test_arrays.npz") as arrays:
        for row in result["fits"]:
            key = f"{row['kind']}-s{row['seed']}"
            assert abs(arrays[f"{key}-nll"].mean() - row["test"]["nll"]) < 1e-12
            assert abs(arrays[f"{key}-top1"].mean() - row["test"]["top1"]) < 1e-12
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    labels = {
        "coherent": "Trainable Hamiltonian",
        "dephased": "Retrained dephased model",
        "mlp": "Classical MLP (similar size)",
        "calibrated_bigram": "Calibrated bigram (all train counts)",
    }
    colors = ["#146a8a", "#bc6632", "#427a45", "#805096"]
    for kind, color in zip(KINDS, colors, strict=True):
        rows = [row for row in lock["fits"] if row["job"]["kind"] == kind]
        for axis, metric in zip(axes, ("nll", "known_top1"), strict=True):
            values = np.array(
                [[item["validation"][metric] for item in row["history"]] for row in rows]
            )
            if metric == "known_top1":
                values *= 100
            epochs = np.arange(values.shape[1])
            axis.plot(epochs, values.mean(0), label=labels[kind], color=color)
            axis.fill_between(epochs, values.min(0), values.max(0), color=color, alpha=0.12)
            axis.set_xlabel("Training epoch")
            axis.grid(alpha=0.18)
    axes[0].set(
        title="Tatoeba · 256-word vocabulary · 8 qubits", ylabel="Validation NLL (lower is better)"
    )
    axes[1].set(title="Known-word targets only", ylabel="Validation top-1 accuracy (%)")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "learning.png", dpi=160)
    plt.close(fig)
    audit = {
        "confirmation_fits_checked": len(lock["fits"]),
        "source_data_weights_selection_hashes": "passed",
        "paired_hamiltonian_initialization": "passed",
        "saved_test_metric_consistency": "passed",
    }
    write_json(output / "audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tatoeba.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/language-born"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--vocabulary", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=32768)
    parser.add_argument("--seeds", nargs="+", type=int, default=[61, 62, 63])
    args = parser.parse_args()
    if (
        min(args.epochs, args.workers, args.train_limit) <= 0
        or args.vocabulary < 8
        or args.vocabulary & (args.vocabulary - 1)
    ):
        parser.error("Positive budgets and a power-of-two vocabulary >=8 are required")
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2 or 11 in args.seeds:
        parser.error("At least two distinct confirmation seeds, excluding screen seed 11")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    started = time.monotonic()
    with new_run_directory(
        args.output, "tatoeba", {"device": "cpu", "torch": torch.__version__}
    ) as output:
        sources = [
            Path(__file__),
            Path("experiments/hamiltonian_study/born_evidence.py"),
            Path("experiments/hamiltonian_study/data.py"),
            Path("experiments/hamiltonian_study/run.py"),
            Path("src/qcse/data.py"),
            Path("src/qcse/outputs.py"),
            Path("uv.lock"),
        ]
        hashes = {}
        for source in sources:
            relative = source.resolve().relative_to(Path.cwd()).as_posix()
            hashes[relative] = digest(source)
            snapshot = output / "source" / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(source.read_bytes())
        manifest = prepare(
            args.data,
            output,
            vocabulary_size=args.vocabulary,
            window=4,
            train_limit=args.train_limit,
            val_limit=4096,
            test_limit=8192,
            split_seed=20261026,
        )
        protocol = {
            "source_sha256": hashes,
            "dataset_sha256": digest(output / "dataset.npz"),
            "manifest": manifest,
            "epochs": args.epochs,
            "batch_size": 512,
            "optimizer": "Adam, no weight decay, norm clipping 5; float64/complex128",
            "encoding": "normalized sum sqrt(0.7**age) exp(i pi age/2) |word>; window 4",
            "readout": "All joint word-basis Born probabilities; no head, prior or postselection",
            "kinds": KINDS,
            "seeds": args.seeds,
            "screen_seed": 11,
            "learning_rates": [0.001, 0.003, 0.01],
            "selection": "Minimum full validation NLL including epoch zero",
            "hypotheses": [
                "Hamiltonian-only training improves held-out next-word NLL and accuracy",
                "Preserving coherence improves on separately retrained dephasing",
            ],
            "pilots": "Validation-only seed-11 pilot informed budgets; test unscored",
            "numpy": np.__version__,
            "workers": args.workers,
        }
        write_json(output / "protocol.json", protocol)
        screen_jobs = [
            Job(str(output), str(output / f"screen-{kind}-{lr}"), kind, 11, lr, args.epochs, 512)
            for kind in KINDS
            for lr in protocol["learning_rates"]
        ]
        screen = execute(screen_jobs, args.workers)
        recipes = {
            kind: min(
                [row for row in screen if row["job"]["kind"] == kind],
                key=lambda row: (row["selected_validation"]["nll"], row["job"]["lr"]),
            )["job"]["lr"]
            for kind in KINDS
        }
        write_json(output / "recipe_lock.json", {"recipes": recipes, "screen": screen})
        confirm_jobs = [
            Job(
                str(output),
                str(output / f"confirm-{kind}-s{seed}"),
                kind,
                seed,
                recipes[kind],
                args.epochs,
                512,
            )
            for kind in KINDS
            for seed in args.seeds
        ]
        fits = execute(confirm_jobs, args.workers)
        write_json(
            output / "selection_lock.json",
            protocol
            | {
                "fits": fits,
                "recipes": recipes,
                "recipe_lock_sha256": digest(output / "recipe_lock.json"),
            },
        )
        result = evaluate(output)
        analyze(output)
        write_json(output / "summary.json", result | {"wall_seconds": time.monotonic() - started})
        print(
            json.dumps(
                {
                    "summary": result["summary"],
                    "baselines": result["baselines"],
                    "intervals": result["intervals"],
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
