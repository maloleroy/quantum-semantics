"""Quiet, resumable jobs. Test evaluation is a separate, explicitly locked stage."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import prepare, transition_probabilities, write_json
from .models import (
    ClassicalEmbedding,
    HamiltonianModel,
    HamiltonianQCSEModel,
    QCSEModel,
    context_distribution,
)

METHODS = [
    "bag",
    "walk",
    "static",
    "predictive",
    "predictive_residual",
    "ordered",
    "directed",
    "no_graph",
    "shuffled",
    "diffusion",
    "directed_diffusion",
    "embedding",
    "qcse_bce",
    "qcse_nll",
    "qcse_born",
    "causal_hamiltonian",
]


def metrics(probabilities, targets):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(targets)
    loss = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-30))
    # Avoid argmax-ID artifacts at ties: fractional credit within tied maxima.
    maximum = p.max(1, keepdims=True)
    top1 = (p[np.arange(len(y)), y] == maximum[:, 0]) / (p == maximum).sum(1)
    threshold = np.partition(p, -min(5, p.shape[1]), axis=1)[:, -min(5, p.shape[1])]
    above = (p > threshold[:, None]).sum(1)
    tied = (p == threshold[:, None]).sum(1)
    py = p[np.arange(len(y)), y]
    top5 = (py > threshold).astype(float) + (py == threshold) * (
        np.minimum(5, p.shape[1]) - above
    ) / tied
    result = {
        "nll": float(loss.mean()),
        "perplexity": float(np.exp(loss.mean())),
        "top1": float(top1.mean()),
        "top5": float(top5.mean()),
        "known_nll": float(loss[y != 0].mean()),
        "known_top1": float(top1[y != 0].mean()),
        "unk_rate": float(np.mean(y == 0)),
        "count": len(y),
    }
    return result, {"nll": loss, "top1": top1, "top5": top5, "targets": y}


def make_model(data, config):
    method, layers = config["method"], config["layers"]
    if method == "embedding":
        return ClassicalEmbedding(len(data["unigram"]), data["train_contexts"].shape[1], layers)
    if method == "causal_hamiltonian":
        return HamiltonianQCSEModel(
            data, layers, config["evolution_time"], config["graph"], config["neighbors"]
        )
    if method.startswith("qcse"):
        return QCSEModel(len(data["unigram"]), layers, born_readout=method == "qcse_born")
    return HamiltonianModel(data, method, layers, config["graph"], config["neighbors"])


def prediction(model, inputs, batch=256, aux=False):
    outputs, successes = [], []
    with torch.no_grad():
        for i in range(0, len(inputs), batch):
            x = inputs[i : i + batch]
            if aux and isinstance(model, HamiltonianModel):
                p, details = model(x, return_aux=True)
                successes.extend(details["success"].tolist())
            else:
                p = model(x)
            outputs.append(p.numpy())
    return np.concatenate(outputs), successes


def job_key(config):
    return f"{config['method']}_{config['graph']}_L{config['layers']}_s{config['seed']}"


def train_one(config):
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    output = Path(config["output"])
    key = job_key(config)
    record_path = output / f"{key}.json"
    if record_path.exists():
        old = json.loads(record_path.read_text())
        if old["config"] != config:
            raise ValueError(f"Refusing stale cache: {record_path}")
        return {"job": key, "cached": True}
    start = time.perf_counter()
    data = dict(np.load(Path(config["dataset"]) / "dataset.npz"))
    model = make_model(data, config)
    inputs = {s: torch.tensor(data[f"{s}_contexts"]) for s in ["train", "validation"]}
    if isinstance(model, QCSEModel):
        inputs = {s: model.encode(x) for s, x in inputs.items()}
    targets = {s: torch.tensor(data[f"{s}_targets"]) for s in inputs}
    method = config["method"]
    lr = 0.003 if method == "embedding" else 0.01 if isinstance(model, QCSEModel) else 0.03
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(config["seed"])
    p, _ = prediction(model, inputs["validation"])
    initial, _ = metrics(p, targets["validation"])
    history = [{"epoch": 0, "validation_nll": initial["nll"]}]
    best_loss, best_epoch = initial["nll"], 0
    best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    gradient_norms = []
    for epoch in range(1, config["epochs"] + 1):
        for ids in np.array_split(
            rng.permutation(len(inputs["train"])),
            int(np.ceil(len(inputs["train"]) / config["batch_size"])),
        ):
            optimizer.zero_grad()
            if method == "qcse_bce":
                assert isinstance(model, QCSEModel)
                _, aux = model(inputs["train"][ids], return_aux=True)
                loss = F.binary_cross_entropy(aux["bits"], model.bits[targets["train"][ids]])
            else:
                p = model(inputs["train"][ids])
                loss = (
                    -p[torch.arange(len(ids)), targets["train"][ids]].clamp_min(1e-30).log().mean()
                )
            if isinstance(model, QCSEModel):
                loss = loss + 0.001 * model.weights.square().sum()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if epoch == 1:
                gradient_norms.append(float(norm))
            optimizer.step()
        if epoch % config["eval_every"] == 0 or epoch == config["epochs"]:
            p, _ = prediction(model, inputs["validation"])
            score, _ = metrics(p, targets["validation"])
            history.append({"epoch": epoch, "validation_nll": score["nll"]})
            if score["nll"] < best_loss:
                best_loss, best_epoch = score["nll"], epoch
                best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    p, success = prediction(model, inputs["validation"], aux=True)
    score, arrays = metrics(p, targets["validation"])
    arrays["groups"] = data["validation_groups"]
    np.savez_compressed(output / f"{key}_validation.npz", **arrays)
    # Only trained parameters; deterministic buffers are reconstructed from the dataset.
    parameters = {k: v.detach() for k, v in model.named_parameters()}
    torch.save(parameters, output / f"{key}.pt")
    record = {
        "config": config,
        "parameters": sum(p.numel() for p in model.parameters()),
        "selected_epoch": best_epoch,
        "validation": score,
        "history": history,
        "initial_gradient_norm_mean": float(np.mean(gradient_norms)),
        "seconds": time.perf_counter() - start,
    }
    if isinstance(model, HamiltonianModel):
        record["unigram_mixture"] = float(model.mixture.detach().sigmoid())
        record["success_mean"] = float(np.mean(success))
    write_json(record_path, record)
    return {"job": key, "validation_nll": best_loss, "seconds": round(record["seconds"], 1)}


def baseline_predictions(data, contexts, method, strength=10.0, decay=0.7):
    u = data["unigram"]
    if method == "unigram":
        return np.broadcast_to(u, (len(contexts), len(u))).copy()
    p = transition_probabilities(data["transitions"], u, strength)
    if method == "bigram":
        return p[contexts[:, -1]]
    c = context_distribution(torch.tensor(contexts), len(u), decay).numpy()
    return c @ p


def fit_baselines(dataset):
    torch.set_num_threads(1)
    directory = Path(dataset)
    data = dict(np.load(directory / "dataset.npz"))
    result = {}
    for method in ["unigram", "bigram", "pooled_bigram"]:
        candidates = []
        for strength in [10.0] if method == "unigram" else [0.1, 1.0, 10.0, 100.0]:
            for decay in [0.7] if method != "pooled_bigram" else [0.3, 0.7, 1.0]:
                p = baseline_predictions(data, data["validation_contexts"], method, strength, decay)
                score, _ = metrics(p, data["validation_targets"])
                candidates.append({"strength": strength, "decay": decay, "validation": score})
        selected = min(candidates, key=lambda c: c["validation"]["nll"])
        result[method] = selected | {"candidates": candidates}
    write_json(directory / "baselines.json", result)
    return {m: r["validation"]["nll"] for m, r in result.items()}


def evaluate(output, dataset, lock):
    """Evaluate only jobs listed in an immutable selection manifest, saved beforehand."""
    torch.set_num_threads(1)
    output, dataset = Path(output), Path(dataset)
    selection = json.loads(Path(lock).read_text())
    data = dict(np.load(dataset / "dataset.npz"))
    contexts = torch.tensor(data["test_contexts"])
    results = {}
    for key in selection["jobs"]:
        record = json.loads((output / f"{key}.json").read_text())
        model = make_model(data, record["config"])
        model.load_state_dict(torch.load(output / f"{key}.pt", weights_only=True), strict=False)
        inputs = model.encode(contexts) if isinstance(model, QCSEModel) else contexts
        p, success = prediction(model, inputs, aux=True)
        result, arrays = metrics(p, data["test_targets"])
        arrays["groups"] = data["test_groups"]
        if success:
            result["success_mean"] = float(np.mean(success))
            result["success_inverse_mean"] = float(np.mean(1 / np.maximum(success, 1e-30)))
        np.savez_compressed(output / f"{key}_test.npz", **arrays)
        results[key] = result
    baselines = json.loads((dataset / "baselines.json").read_text())
    for method, selected in baselines.items():
        p = baseline_predictions(
            data, data["test_contexts"], method, selected["strength"], selected["decay"]
        )
        result, arrays = metrics(p, data["test_targets"])
        arrays["groups"] = data["test_groups"]
        np.savez_compressed(output / f"{method}_test.npz", **arrays)
        results[method] = result
    write_json(output / "test_results.json", {"selection": selection, "metrics": results})
    return {k: round(v["nll"], 4) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "baselines", "evaluate"])
    parser.add_argument("--data", type=Path, default=Path("data/tatoeba.csv"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("experiments/hamiltonian_study/results/data")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/hamiltonian_study/results/screen")
    )
    parser.add_argument("--vocabulary", type=int, default=512)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=8192)
    parser.add_argument("--val-limit", type=int, default=4096)
    parser.add_argument("--test-limit", type=int, default=8192)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["walk"])
    parser.add_argument("--graphs", nargs="+", choices=["count", "ppmi", "npmi"], default=["count"])
    parser.add_argument("--layers", nargs="+", type=int, default=[16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=24)
    parser.add_argument("--evolution-time", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        print(
            json.dumps(
                prepare(
                    args.data,
                    args.dataset,
                    args.vocabulary,
                    args.window,
                    args.train_limit,
                    args.val_limit,
                    args.test_limit,
                )
            )
        )
    elif args.stage == "baselines":
        print(fit_baselines(args.dataset))
    elif args.stage == "evaluate":
        if args.lock is None:
            parser.error("evaluate requires --lock, saved before test evaluation")
        print(evaluate(args.output, args.dataset, args.lock))
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256((args.dataset / "dataset.npz").read_bytes()).hexdigest()
        code_digest = hashlib.sha256(
            b"".join(
                p.read_bytes()
                for p in [
                    Path(__file__).parent / name for name in ("data.py", "models.py", "run.py")
                ]
            )
        ).hexdigest()
        common = {
            "dataset": str(args.dataset),
            "output": str(args.output),
            "dataset_sha256": digest,
            "code_sha256": code_digest,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_every": args.eval_every,
            "neighbors": args.neighbors,
        }
        configs = [
            common | {"method": m, "graph": g, "layers": d, "seed": s}
            for m in args.methods
            for g in args.graphs
            for d in args.layers
            for s in args.seeds
        ]
        for config in configs:
            if config["method"] == "causal_hamiltonian":
                config["evolution_time"] = args.evolution_time
        environment = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "workers": args.workers,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "numerical_threads": 1,
            "cpu_count": os.cpu_count(),
        }
        write_json(args.output / f"environment_{int(time.time())}.json", environment)
        if args.workers == 1:
            for config in configs:
                print(train_one(config), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for result in executor.map(train_one, configs):
                    print(result, flush=True)


if __name__ == "__main__":
    main()
