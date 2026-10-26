"""Train Hamiltonians with fixed inputs and Born readout; lock before testing."""

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from qcse.outputs import new_run_directory, save_npz, write_json

KINDS = ("coherent", "dephased", "linear", "mlp", "wide_mlp")
UCI_SOURCE = (
    "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"
)


@dataclass(frozen=True)
class Spec:
    task: str
    dimension: int
    classes: int = 16


@dataclass(frozen=True)
class Job:
    dataset: str
    output: str
    task: str
    kind: str
    seed: int
    lr: float
    epochs: int
    batch_size: int


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hermitian(coefficients: torch.Tensor, dimension: int) -> torch.Tensor:
    """Exactly d**2 real coordinates: diagonal, real upper, imaginary upper."""
    rows, columns = torch.triu_indices(dimension, dimension, 1)
    count = len(rows)
    values = torch.complex(
        coefficients[dimension : dimension + count], coefficients[dimension + count :]
    )
    upper = torch.zeros((dimension, dimension), dtype=torch.complex128)
    upper = upper.index_put((rows, columns), values)
    return torch.diag(coefficients[:dimension]).to(torch.complex128) + upper + upper.mH


def phase_table(dimension: int) -> torch.Tensor:
    ids = torch.arange(dimension, dtype=torch.float64)
    return torch.exp(2j * torch.pi * ids[:, None] * ids[None, :] / dimension)


def quantum_probabilities(
    inputs: torch.Tensor, unitary: torch.Tensor, spec: Spec, dephased: bool = False
) -> torch.Tensor:
    if spec.task == "modular":
        if dephased:
            # Dephase between U and the fixed input phase. D_b then has no effect.
            transition = unitary.abs().square()
            return transition[:, inputs[:, 0]].T @ transition
        state = unitary[:, inputs[:, 0]].T * phase_table(spec.dimension)[inputs[:, 1]]
        return (state @ unitary.conj()).abs().square()
    if dephased:
        probabilities = inputs.square() @ unitary.abs().square().T
    else:
        probabilities = (inputs.to(torch.complex128) @ unitary.T).abs().square()
    # Four leading readout qubits; sum all outcomes of the other two. No conditioning.
    return probabilities.reshape(-1, spec.classes, spec.dimension // spec.classes).sum(-1)


class BornModel(nn.Module):
    def __init__(self, spec: Spec, kind: str):
        super().__init__()
        self.spec, self.kind = spec, kind
        self.coefficients = nn.Parameter(
            torch.randn(spec.dimension**2, dtype=torch.float64)
            * (0.15 if spec.task == "modular" else 0.03)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        unitary = torch.matrix_exp(-1j * hermitian(self.coefficients, self.spec.dimension))
        return quantum_probabilities(inputs, unitary, self.spec, self.kind == "dephased")


class ClassicalModel(nn.Module):
    def __init__(self, spec: Spec, kind: str):
        super().__init__()
        self.spec = spec
        width = 2 * spec.dimension if spec.task == "modular" else spec.dimension
        hidden = (
            (5 if spec.task == "modular" else 50)
            if kind == "mlp"
            else (32 if spec.task == "modular" else 256)
        )
        self.network = (
            nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, spec.classes))
            if kind != "linear"
            else nn.Linear(width, spec.classes)
        ).double()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = (
            F.one_hot(inputs, self.spec.dimension).flatten(1).double()
            if self.spec.task == "modular"
            else inputs
        )
        return self.network(features).softmax(-1)


def make_model(spec: Spec, kind: str) -> BornModel | ClassicalModel:
    return BornModel(spec, kind) if kind in {"coherent", "dephased"} else ClassicalModel(spec, kind)


def normalize_images(pixels: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(pixels, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero image has no amplitude encoding")
    return pixels.astype(np.float64) / norms


def prepare(output: Path, archive: Path) -> dict:
    rng = np.random.default_rng(314159)
    parts: dict[str, list[tuple[int, int]]] = {"train": [], "validation": [], "test": []}
    for first in range(16):
        second = rng.permutation(16)
        for name, ids in zip(parts, (second[:10], second[10:13], second[13:]), strict=True):
            parts[name].extend((first, int(value)) for value in ids)
    symbolic = {}
    for name, pairs in parts.items():
        inputs = np.array(pairs, dtype=np.int64)
        symbolic[f"{name}_inputs"] = inputs
        symbolic[f"{name}_targets"] = inputs.sum(1) % 16
    save_npz(output / "modular-fit.npz", **{k: v for k, v in symbolic.items() if "test" not in k})
    save_npz(output / "modular-test.npz", **{k: v for k, v in symbolic.items() if "test" in k})

    with zipfile.ZipFile(archive) as source:
        raw_train, raw_test = source.read("optdigits.tra"), source.read("optdigits.tes")
    training = np.loadtxt(io.BytesIO(raw_train), delimiter=",", dtype=np.int64)
    testing = np.loadtxt(io.BytesIO(raw_test), delimiter=",", dtype=np.int64)
    train_hashes = np.array([hashlib.sha256(row[:64].tobytes()).hexdigest() for row in training])
    test_hashes = {hashlib.sha256(row[:64].tobytes()).hexdigest() for row in testing}
    if set(train_hashes) & test_hashes:
        raise ValueError("Exact images overlap the official train/test partitions")
    rng = np.random.default_rng(271828)
    validation_groups: list[str] = []
    for label in range(10):
        groups = np.unique(train_hashes[training[:, -1] == label])
        validation_groups.extend(rng.permutation(groups)[: len(groups) // 5])
    validation = np.isin(train_hashes, validation_groups)
    inputs = normalize_images(training[:, :64])
    save_npz(
        output / "digits-fit.npz",
        train_inputs=inputs[~validation],
        train_targets=training[~validation, -1],
        validation_inputs=inputs[validation],
        validation_targets=training[validation, -1],
        train_indices=np.flatnonzero(~validation),
        validation_indices=np.flatnonzero(validation),
    )
    save_npz(
        output / "digits-test.npz",
        test_inputs=normalize_images(testing[:, :64]),
        test_targets=testing[:, -1],
    )
    provenance = {
        "modular": {
            "rule": "target = (a + b) mod 16",
            "split_seed": 314159,
            "split_sizes": {key: len(value) for key, value in parts.items()},
        },
        "digits": {
            "source": UCI_SOURCE,
            "archive_sha256": digest(archive),
            "train_file_sha256": hashlib.sha256(raw_train).hexdigest(),
            "test_file_sha256": hashlib.sha256(raw_test).hexdigest(),
            "validation_split_seed": 271828,
            "split_sizes": {
                "train": int((~validation).sum()),
                "validation": int(validation.sum()),
                "test": len(testing),
            },
            "exact_image_groups_disjoint": True,
        },
        "files": {path.name: digest(path) for path in output.glob("*.npz")},
    }
    write_json(output / "data.json", provenance)
    return provenance


def load_fit(dataset: Path, task: str) -> tuple[torch.Tensor, ...]:
    with np.load(dataset / f"{task}-fit.npz") as source:
        return tuple(
            torch.tensor(source[f"{part}_{field}"])
            for part in ("train", "validation")
            for field in ("inputs", "targets")
        )


def metrics(probabilities: torch.Tensor, targets: torch.Tensor) -> dict:
    correct = probabilities[torch.arange(len(targets)), targets]
    if not torch.isfinite(probabilities).all() or torch.min(probabilities) < 0:
        raise ValueError("Invalid probabilities")
    norm_error = float((probabilities.sum(-1) - 1).abs().max())
    if norm_error > 1e-10:
        raise ValueError(f"Born probability normalization error: {norm_error}")
    return {
        "nll": float(-correct.clamp_min(1e-15).log().mean()),
        "top1": float((probabilities.argmax(-1) == targets).double().mean()),
        "norm_error": norm_error,
    }


def fit(job: Job) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(job.seed)
    spec = Spec(job.task, 16 if job.task == "modular" else 64)
    inputs, targets, val_inputs, val_targets = load_fit(Path(job.dataset), job.task)
    model = make_model(spec, job.kind)
    output = Path(job.output)
    output.mkdir()
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    parameter_hashes = {
        name: hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
        for name, value in model.named_parameters()
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=job.lr)
    rng = np.random.default_rng(job.seed)
    best_nll = float("inf")
    best_epoch, best = 0, initial
    history = []
    started = time.monotonic()
    first_gradient = None
    for epoch in range(job.epochs + 1):
        with torch.no_grad():
            validation = metrics(model(val_inputs), val_targets)
        history.append({"epoch": epoch, "validation": validation})
        if validation["nll"] < best_nll:
            best_nll, best_epoch = validation["nll"], epoch
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch == job.epochs:
            break
        order = rng.permutation(len(targets))
        total_loss = 0.0
        for start in range(0, len(order), job.batch_size):
            ids = order[start : start + job.batch_size]
            optimizer.zero_grad()
            probabilities = model(inputs[ids])
            loss = (
                -probabilities[torch.arange(len(ids)), targets[ids]].clamp_min(1e-15).log().mean()
            )
            loss.backward()
            if first_gradient is None:
                first_gradient = (
                    sum(
                        float(value.grad.square().sum())
                        for value in model.parameters()
                        if value.grad is not None
                    )
                    ** 0.5
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
        history[-1]["next_epoch_online_train_nll"] = total_loss / len(order)
    model.load_state_dict(best)
    with torch.no_grad():
        selected_train = metrics(model(inputs), targets)
        selected_validation = metrics(model(val_inputs), val_targets)
    torch.save(
        {"spec": asdict(spec), "kind": job.kind, "state": best, "initial": initial},
        output / "weights.pt",
    )
    result = {
        "job": asdict(job),
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_parameter_hashes": parameter_hashes,
        "selected_epoch": best_epoch,
        "initial_validation": history[0]["validation"],
        "selected_validation": selected_validation,
        "selected_training": selected_train,
        "first_gradient_norm": first_gradient,
        "history": history,
        "weights_sha256": digest(output / "weights.pt"),
        "seconds": time.monotonic() - started,
    }
    write_json(output / "result.json", result)
    return result


def configure_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def execute(jobs: list[Job], workers: int) -> list[dict]:
    # A fresh pool per stage; each process runs several jobs with one numeric thread.
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=configure_worker,
    ) as pool:
        records = []
        for record in pool.map(fit, jobs):
            records.append(record)
            job = record["job"]
            print(
                f"{job['task']} {job['kind']} s{job['seed']} lr={job['lr']}: "
                f"val {record['selected_validation']['nll']:.5f}, "
                f"accuracy {record['selected_validation']['top1']:.2%}",
                flush=True,
            )
    return records


def select_recipes(records: list[dict]) -> dict[str, dict[str, float]]:
    recipes: dict[str, dict[str, float]] = {}
    for task in ("modular", "digits"):
        recipes[task] = {}
        for kind in KINDS:
            rows = [
                row for row in records if row["job"]["task"] == task and row["job"]["kind"] == kind
            ]
            selected = min(
                rows, key=lambda row: (row["selected_validation"]["nll"], row["job"]["lr"])
            )
            recipes[task][kind] = selected["job"]["lr"]
    return recipes


def paired_interval(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(161803)
    means = np.array(
        [values[rng.integers(len(values), size=len(values))].mean() for _ in range(2000)]
    )
    return np.quantile(means, [0.025, 0.975]).tolist()


def evaluate(output: Path) -> dict:
    """The first entry point that scores held-out test examples; requires a lock."""
    lock = json.loads((output / "selection_lock.json").read_text())
    if (output / "test_results.json").exists():
        raise ValueError("Test already evaluated; use the saved results")
    if digest(Path(__file__)) != lock["source_sha256"]:
        raise ValueError("Source changed after selection lock")
    for name, expected in lock["data"]["files"].items():
        if digest(output / name) != expected:
            raise ValueError("Data changed after selection lock")
    results = []
    predictions: dict[str, np.ndarray] = {}
    for record in lock["fits"]:
        job = record["job"]
        directory = Path(job["output"])
        if digest(directory / "weights.pt") != record["weights_sha256"]:
            raise ValueError("Weights changed after selection lock")
        spec = Spec(job["task"], 16 if job["task"] == "modular" else 64)
        with np.load(output / f"{spec.task}-test.npz") as data:
            inputs, targets = torch.tensor(data["test_inputs"]), torch.tensor(data["test_targets"])
        saved = torch.load(directory / "weights.pt", weights_only=True)
        model = make_model(spec, job["kind"])
        model.load_state_dict(saved["state"])
        with torch.no_grad():
            probabilities = model(inputs)
            selected = metrics(probabilities, targets)
            model.load_state_dict(saved["initial"])
            initial = metrics(model(inputs), targets)
        key = f"{spec.task}-{job['kind']}-s{job['seed']}"
        predictions[key] = probabilities.numpy()
        results.append(
            {
                "task": spec.task,
                "kind": job["kind"],
                "seed": job["seed"],
                "selected_epoch": record["selected_epoch"],
                "initial_test": initial,
                "test": selected,
                "parameters": record["parameters"],
            }
        )
    summary = {}
    for task in ("modular", "digits"):
        summary[task] = {}
        for kind in KINDS:
            rows = [row for row in results if row["task"] == task and row["kind"] == kind]
            summary[task][kind] = {
                "parameters": rows[0]["parameters"],
                "nll_mean": float(np.mean([row["test"]["nll"] for row in rows])),
                "nll_sd": float(np.std([row["test"]["nll"] for row in rows], ddof=1)),
                "top1_mean": float(np.mean([row["test"]["top1"] for row in rows])),
                "top1_sd": float(np.std([row["test"]["top1"] for row in rows], ddof=1)),
                "initial_top1_mean": float(np.mean([row["initial_test"]["top1"] for row in rows])),
                "initial_nll_mean": float(np.mean([row["initial_test"]["nll"] for row in rows])),
            }
        with np.load(output / f"{task}-test.npz") as data:
            targets = data["test_targets"]
        differences = []
        for seed in lock["confirmation_seeds"]:
            coherent = predictions[f"{task}-coherent-s{seed}"][np.arange(len(targets)), targets]
            dephased = predictions[f"{task}-dephased-s{seed}"][np.arange(len(targets)), targets]
            differences.append(
                -np.log(np.maximum(coherent, 1e-15)) + np.log(np.maximum(dephased, 1e-15))
            )
        per_example = np.mean(differences, axis=0)
        summary[task]["coherent_minus_dephased_nll"] = {
            "mean": float(per_example.mean()),
            "example_bootstrap_95": paired_interval(per_example),
            "scope": "Conditional on this split and fitted seeds; not writer/split uncertainty",
        }
    save_npz(output / "test_predictions.npz", **predictions)
    result = {
        "selection_lock_sha256": digest(output / "selection_lock.json"),
        "fits": results,
        "summary": summary,
    }
    write_json(output / "test_results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/born-evidence"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--modular-epochs", type=int, default=250)
    parser.add_argument("--digits-epochs", type=int, default=60)
    parser.add_argument("--confirmation-seeds", nargs="+", type=int, default=[47, 48, 49, 50, 51])
    args = parser.parse_args()
    if min(args.workers, args.modular_epochs, args.digits_epochs) <= 0:
        parser.error("Workers and epoch budgets must be positive")
    if (
        len(args.confirmation_seeds) < 2
        or 11 in args.confirmation_seeds
        or len(set(args.confirmation_seeds)) != len(args.confirmation_seeds)
    ):
        parser.error("At least two distinct confirmation seeds, different from screen seed 11")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    started = time.monotonic()
    with new_run_directory(
        args.output, "fixed-encoding", {"device": "cpu", "torch": torch.__version__}
    ) as output:
        source_hash = digest(Path(__file__))
        (output / "executed_source.py").write_bytes(Path(__file__).read_bytes())
        (output / "uv.lock").write_bytes(Path("uv.lock").read_bytes())
        data = prepare(output, args.digits_archive)
        protocol = {
            "source_sha256": source_hash,
            "uv_lock_sha256": digest(output / "uv.lock"),
            "numpy": np.__version__,
            "data": data,
            "screen_seed": 11,
            "confirmation_seeds": args.confirmation_seeds,
            "kinds": KINDS,
            "selection": "Minimum validation NLL including epoch zero; no test tuning",
            "hypotheses": [
                "Hamiltonian-only training beats its frozen initialization on unseen data",
                "Coherent evolution beats a separately retrained, equally parameterized "
                "dephased model",
            ],
            "limits": "Not a quantum computational advantage claim or a language benchmark",
            "optimizer": "Adam; no decay; norm clipping 5; float64/complex128",
            "epochs": {"modular": args.modular_epochs, "digits": args.digits_epochs},
            "batch_size": {"modular": 160, "digits": 256},
            "lr_grids": {
                "modular_quantum": [0.01, 0.03, 0.1],
                "digits_quantum": [0.003, 0.01, 0.03],
                "classical": [0.003, 0.01, 0.03],
            },
            "workers": args.workers,
        }
        write_json(output / "protocol.json", protocol)
        jobs = []
        for task, epochs, batch_size in (
            ("modular", args.modular_epochs, 160),
            ("digits", args.digits_epochs, 256),
        ):
            for kind in KINDS:
                grid = (
                    [0.01, 0.03, 0.1]
                    if task == "modular" and kind in {"coherent", "dephased"}
                    else [0.003, 0.01, 0.03]
                )
                for lr in grid:
                    jobs.append(
                        Job(
                            str(output),
                            str(output / f"screen-{task}-{kind}-{lr}"),
                            task,
                            kind,
                            11,
                            lr,
                            epochs,
                            batch_size,
                        )
                    )
        screen = execute(jobs, args.workers)
        recipes = select_recipes(screen)
        write_json(output / "recipe_lock.json", {"recipes": recipes, "screen": screen})
        confirmations = []
        for task, epochs, batch_size in (
            ("modular", args.modular_epochs, 160),
            ("digits", args.digits_epochs, 256),
        ):
            for kind in KINDS:
                for seed in args.confirmation_seeds:
                    confirmations.append(
                        Job(
                            str(output),
                            str(output / f"confirm-{task}-{kind}-s{seed}"),
                            task,
                            kind,
                            seed,
                            recipes[task][kind],
                            epochs,
                            batch_size,
                        )
                    )
        confirmed = execute(confirmations, args.workers)
        write_json(
            output / "selection_lock.json",
            protocol
            | {
                "fits": confirmed,
                "recipes": recipes,
                "recipe_lock_sha256": digest(output / "recipe_lock.json"),
            },
        )
        result = evaluate(output)
        write_json(
            output / "summary.json",
            {"summary": result["summary"], "wall_seconds": time.monotonic() - started},
        )
        print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
