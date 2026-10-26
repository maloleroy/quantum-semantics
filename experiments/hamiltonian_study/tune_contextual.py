"""Staged, paired CPU tuning of the contextual Hamiltonian and classical control."""

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from qcse.outputs import new_run_directory, save_npz, write_json

from .contextual import ContextData, ResidualConfig, fit, load_data, tensor_hash, toy_data

MODELS = ("hamiltonian", "classical")
WORKER_DATA: ContextData | None = None


@dataclass(frozen=True)
class FitJob:
    stage: str
    kind: str
    lr: float
    weight_decay: float
    seed: int


def job_name(job: FitJob) -> str:
    return f"{job.kind}-lr{job.lr:g}-wd{job.weight_decay:g}-s{job.seed}"


def initialize_worker(data: ContextData, threads: int):
    global WORKER_DATA
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    WORKER_DATA = data


def run_job(job: FitJob, config: ResidualConfig, epochs: int, batch: int, root: str) -> dict:
    assert WORKER_DATA is not None
    output = Path(root)
    lock_hash = None
    if job.stage == "confirmation":
        lock = output / "selection_lock.json"
        selected = json.loads(lock.read_text())["selected"][job.kind]
        assert any(
            row["lr"] == job.lr and row["weight_decay"] == job.weight_decay for row in selected
        )
        lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest()
    folder = output / job.stage / job_name(job)
    folder.mkdir(parents=True, exist_ok=False)
    result = fit(
        WORKER_DATA,
        config,
        job.kind,
        job.seed,
        epochs,
        batch,
        job.lr,
        folder,
        job.weight_decay,
    )
    record = {
        "job": asdict(job),
        "directory": str(folder.relative_to(output)),
        "selection_lock_sha256": lock_hash,
        "result": result,
    }
    write_json(folder / "job.json", record)
    return record


def select_recipes(records: list[dict], keep: int) -> dict[str, list[dict]]:
    selected = {}
    for kind in MODELS:
        rows = [row for row in records if row["job"]["kind"] == kind]
        ranked = sorted(
            rows,
            key=lambda row: (
                row["result"]["selected_validation"]["nll"],
                row["job"]["lr"],
                row["job"]["weight_decay"],
            ),
        )
        if len(ranked) < keep:
            raise ValueError("Not enough complete screening fits for selection")
        selected[kind] = [
            {
                "lr": row["job"]["lr"],
                "weight_decay": row["job"]["weight_decay"],
                "screen_validation_nll": row["result"]["selected_validation"]["nll"],
                "screen_job": row["directory"],
            }
            for row in ranked[:keep]
        ]
    return selected


def run_stage(
    executor: ProcessPoolExecutor,
    jobs: list[FitJob],
    config: ResidualConfig,
    epochs: int,
    batch: int,
    output: Path,
) -> list[dict]:
    futures = {
        executor.submit(run_job, job, config, epochs, batch, str(output)): job for job in jobs
    }
    records = []
    started = time.monotonic()
    for future in as_completed(futures):
        result = future.result()
        records.append(result)
        stage = result["job"]["stage"]
        write_json(output / f"{stage}.json", {"records": records})
        print(
            f"{stage}: {len(records)}/{len(jobs)} fits complete; "
            f"elapsed {time.monotonic() - started:.1f}s; {job_name(futures[future])}",
            flush=True,
        )
    return sorted(records, key=lambda row: row["directory"])


def snapshot_sources(output: Path) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    names = [
        "experiments/hamiltonian_study/tune_contextual.py",
        "experiments/hamiltonian_study/contextual.py",
        "experiments/hamiltonian_study/data.py",
        "experiments/hamiltonian_study/run.py",
        "src/qcse/outputs.py",
        "pyproject.toml",
        "uv.lock",
    ]
    hashes = {}
    for name in names:
        content = (root / name).read_bytes()
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    return hashes


def check_pairing(records: list[dict]):
    hashes = {}
    for row in records:
        seed = row["job"]["seed"]
        initial = row["result"]["initial_parameter_hashes"]
        if seed in hashes:
            assert hashes[seed] == initial
        hashes[seed] = initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--toy", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/contextual-tuning"))
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.001, 0.003, 0.01])
    parser.add_argument("--decays", nargs="+", type=float, default=[0.0, 0.001, 0.01])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--screen-seed", type=int, default=11)
    parser.add_argument("--confirmation-seeds", nargs="+", type=int, default=[47, 48, 49])
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=32768)
    parser.add_argument("--validation-limit", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.workers, args.threads, args.keep, *args.lrs) <= 0:
        parser.error("Epochs, batch, workers, threads, keep and LRs must be positive")
    if min(args.train_limit, args.validation_limit, *args.decays) < 0:
        parser.error("Dataset limits and decay must be nonnegative")
    if len(set(args.lrs)) != len(args.lrs) or len(set(args.decays)) != len(args.decays):
        parser.error("Grid entries must be distinct")
    if args.keep > len(args.lrs) * len(args.decays):
        parser.error("Keep cannot exceed the number of recipes")
    if (
        len(set(args.confirmation_seeds)) != len(args.confirmation_seeds)
        or args.screen_seed in args.confirmation_seeds
    ):
        parser.error("Confirmation seeds must be distinct and exclude the screen seed")
    if args.workers * args.threads > (os.cpu_count() or 1):
        parser.error("Workers multiplied by threads exceeds the logical CPU count")
    torch.set_num_threads(1)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    data = (
        toy_data() if args.toy else load_data(args.dataset, args.train_limit, args.validation_limit)
    )
    config = ResidualConfig(
        len(data.vocabulary), data.train_contexts.shape[1], unknown_id=None if args.toy else 0
    )
    runtime = {
        "device": "cpu",
        "torch": torch.__version__,
        "workers": args.workers,
        "threads_per_worker": args.threads,
        "test_scored": False,
    }
    with new_run_directory(args.output, "paired-search", runtime) as output:
        fingerprints = {
            name: tensor_hash(value)
            for name, value in {
                "train_contexts": data.train_contexts,
                "train_targets": data.train_targets,
                "validation_contexts": data.validation_contexts,
                "validation_targets": data.validation_targets,
                "bigram": data.prior,
            }.items()
        }
        settings = vars(args) | {"dataset": str(args.dataset), "output": str(args.output)}
        source_hashes = snapshot_sources(output)
        write_json(
            output / "protocol.json",
            {
                "settings": settings,
                "model": asdict(config),
                "data": data.provenance,
                "fingerprints": fingerprints,
                "source_sha256": source_hashes,
                "vocabulary": data.vocabulary,
                "test_scored": False,
                "selection": "NLL only; top recipes from one-seed screen locked before fresh seeds",
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
        screen_jobs = [
            FitJob("screen", kind, lr, decay, args.screen_seed)
            for lr in args.lrs
            for decay in args.decays
            for kind in MODELS
        ]
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(data, args.threads),
        ) as executor:
            screen = run_stage(executor, screen_jobs, config, args.epochs, args.batch_size, output)
            check_pairing(screen)
            selected = select_recipes(screen, args.keep)
            write_json(output / "selection_lock.json", {"selected": selected})
            confirmation_jobs = [
                FitJob("confirmation", kind, row["lr"], row["weight_decay"], seed)
                for kind in MODELS
                for row in selected[kind]
                for seed in args.confirmation_seeds
            ]
            confirmation = run_stage(
                executor, confirmation_jobs, config, args.epochs, args.batch_size, output
            )
            check_pairing(confirmation)
        repository = Path(__file__).resolve().parents[2]
        assert {
            name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
            for name in source_hashes
        } == source_hashes
        write_json(
            output / "summary.json",
            {"selected": selected, "screen": screen, "confirmation": confirmation},
        )


if __name__ == "__main__":
    main()
