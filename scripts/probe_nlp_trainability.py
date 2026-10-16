"""Conditional circuit-initialization gradient variance on prepared training examples."""

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from qcse.benchmark import code_hashes, load_manifest, runtime, save_code_snapshot, training_loss
from qcse.benchmark_models import HybridCircuitModel
from qcse.outputs import new_run_directory, save_npz, write_json


def probe(vocabulary, window, contexts, targets, qubits, layers, scale, seeds, device):
    # The same seed and depth-2 template fix encoder/head within each register size.
    torch.manual_seed(619)
    template = HybridCircuitModel(vocabulary, window, dict(qubits=qubits, layers=2))
    common = {k: v for k, v in template.state_dict().items() if not k.startswith("circuit.")}
    h = hashlib.sha256()
    for name, value in sorted(common.items()):
        h.update(name.encode())
        h.update(value.cpu().numpy().tobytes())
    model = HybridCircuitModel(vocabulary, window, dict(qubits=qubits, layers=layers))
    model.load_state_dict(common, strict=False)
    model.to(device).eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "circuit.weights")
    gradients, losses = [], []
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        weights = scale * torch.randn(model.circuit.weights.shape, generator=generator)
        with torch.no_grad():
            model.circuit.weights.copy_(weights.to(device))
        loss = training_loss(model, contexts.to(device), targets.to(device))
        gradient = torch.autograd.grad(loss, model.circuit.weights)[0]
        if not torch.isfinite(gradient).all() or not torch.isfinite(loss):
            raise ValueError("Non-finite NLP probe gradient/loss")
        gradients.append(gradient.detach().cpu().numpy().astype(np.float64))
        losses.append(float(loss.detach()))
    values = np.stack(gradients)
    return dict(
        qubits=qubits,
        layers=layers,
        init_scale=scale,
        gradient_dtype=str(model.circuit.weights.dtype),
        seeds=len(seeds),
        common_state_sha256=h.hexdigest(),
        first_rx_gradient_variance=float(values[:, 0, 0].var(ddof=1)),
        mean_gradient_variance=float(values.var(0, ddof=1).mean()),
        rms_gradient=float(np.sqrt(np.mean(values**2))),
        loss_mean=float(np.mean(losses)),
    ), dict(gradients=values, losses=np.array(losses), seeds=np.array(seeds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--qubits", nargs="+", type=int, default=[2, 4, 8, 12])
    parser.add_argument("--layers", nargs="+", type=int, default=[2, 8])
    parser.add_argument("--init-scales", nargs="+", type=float, default=[0.05, 1.0])
    parser.add_argument("--seeds", type=int, default=32)
    args = parser.parse_args()
    if args.seeds < 2 or not all(2 <= q <= 12 for q in args.qubits) or min(args.layers) < 1:
        parser.error("Need >=2 seeds, 2..12 qubits and positive depths")
    if any(not np.isfinite(v) or v <= 0 for v in args.init_scales):
        parser.error("Initialization scales must be finite and positive")
    if any(
        len(values) != len(set(values)) for values in (args.qubits, args.layers, args.init_scales)
    ):
        parser.error("Grid values must be distinct")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    manifest = load_manifest(args.study.resolve())
    settings = manifest["settings"]
    torch.set_num_threads(settings["threads"])
    torch.use_deterministic_algorithms(True)
    objective = settings["objectives"][0]
    with np.load(args.study / f"{objective}.npz") as archive:
        ids = np.random.default_rng(619).choice(
            archive["train_0"], min(8, len(archive["train_0"])), replace=False
        )
        contexts = torch.as_tensor(archive["contexts"][ids])
        targets = torch.as_tensor(archive["targets"][ids])
    metadata = dict(
        manifest=manifest["fingerprint"],
        objective=objective,
        fold=0,
        runtime=runtime(args.device),
        fixed_non_circuit_seed=619,
        code_sha256=code_hashes(),
    )
    with new_run_directory(args.output, "nlp-gradient-probe", metadata) as output:
        save_code_snapshot(output, metadata["code_sha256"])
        save_npz(output / "inputs.npz", ids=ids, contexts=contexts.numpy(), targets=targets.numpy())
        rows = []
        for q in args.qubits:
            for layers in args.layers:
                for scale in args.init_scales:
                    summary, arrays = probe(
                        manifest["vocabulary_size"],
                        settings["window"],
                        contexts,
                        targets,
                        q,
                        layers,
                        scale,
                        list(range(42, 42 + args.seeds)),
                        args.device,
                    )
                    rows.append(summary)
                    save_npz(output / f"q{q}-depth{layers}-init{scale:g}.npz", **arrays)
                    write_json(
                        output / "summary.json",
                        dict(
                            **metadata,
                            rows=rows,
                            scope="Actual train-fold CE, fixed 8-example batch and encoder/head "
                            "per size; circuit initialization varies. Register size also changes "
                            "encoder/head shape. Conditional initial-gradient variance only, "
                            "not a proof of asymptotic NLP barren plateaus or trained behavior.",
                        ),
                    )
        print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
