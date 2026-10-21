"""Diagnostic microbatch parity/timings on fixed training inputs, both tracks.

Never changes a manifest or selects a training microbatch. Evaluation-mode
parity disables dropout; timed training keeps normal dropout and diagnostics.
"""

import argparse
import copy
import hashlib
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from qcse.benchmark import code_hashes, load_manifest, runtime, training_loss
from qcse.benchmark_models import create_benchmark_model
from qcse.outputs import save_npz, write_json
from qcse.trainability import circuit_gradient_summary

MAIN = ("mlp-32", "bert-32", "hybrid-q8-depth2-init0.05", "hybrid-q8-identity")


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def vector(tensors):
    return torch.cat([t.detach().flatten().cpu() for t in tensors])


def fresh_model(template, device):
    torch.manual_seed(42)
    return copy.deepcopy(template).to(device)


def parity_state(model, contexts, targets, microbatch, lr):
    """One eval-mode forward/gradient/Adam update, including a partial batch."""
    model.eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    optimizer.zero_grad(set_to_none=True)
    logits, total = [], 0.0
    device = next(model.parameters()).device
    for start in range(0, len(targets), microbatch):
        x, y = (
            contexts[start : start + microbatch].to(device),
            targets[start : start + microbatch].to(device),
        )
        predicted = model(x)
        loss = F.cross_entropy(predicted, y)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite pilot loss")
        (loss * len(y) / len(targets)).backward()
        total += float(loss.detach()) * len(y) / len(targets)
        logits.append(predicted.detach().cpu())
    gradients = vector([p.grad for p in parameters if p.grad is not None])
    torch.nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
    optimizer.step()
    moments = vector(
        [
            optimizer.state[p][k]
            for p in parameters
            for k in ("exp_avg", "exp_avg_sq")
            if p in optimizer.state
        ]
    )
    return dict(
        loss=torch.tensor([total]),
        logits=torch.cat(logits),
        gradients=gradients,
        parameters=vector(parameters),
        moments=moments,
    )


def compare(reference, actual):
    checks = {}
    for name in reference:
        a, b = reference[name], actual[name]
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Non-finite parity values")
        atol = 5e-5 if name == "parameters" else 2e-6
        checks[name] = dict(
            max_abs=float((a - b).abs().max()),
            passed=bool(torch.allclose(a, b, atol=atol, rtol=5e-4)),
            atol=atol,
            rtol=5e-4,
        )
    return dict(passed=all(c["passed"] for c in checks.values()), checks=checks)


def measure(model, contexts, targets, microbatch, lr, repeats, device):
    """Match frozen training accumulation/diagnostics/clipping, not corpus fits."""
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    durations = []
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for iteration in range(repeats + 1):
        synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, len(targets), microbatch):
            x = contexts[offset : offset + microbatch].to(device)
            y = targets[offset : offset + microbatch].to(device)
            loss = training_loss(model, x, y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite timed loss")
            (loss * len(y) / len(targets)).backward()
        circuit_gradient_summary(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        synchronize(device)
        if iteration:
            durations.append(time.perf_counter() - start)
    return dict(
        training_seconds=durations,
        median_training_seconds=float(np.median(durations)),
        cuda_peak_allocated=torch.cuda.max_memory_allocated() if device == "cuda" else None,
        cuda_peak_reserved=torch.cuda.max_memory_reserved() if device == "cuda" else None,
    )


def profile(
    parent,
    output,
    device="cuda",
    repeats=5,
    microbatches=(8, 16, 32, 64),
    batch_sizes=(64, 32),
    names=MAIN,
    threads=4,
):
    if min(repeats, threads, *microbatches, *batch_sizes) < 1:
        raise ValueError("Pilot dimensions/repeats must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)
    output.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter()
    report: dict = dict(
        status="running",
        device=device,
        runtime=runtime(device),
        repeats=repeats,
        started_at=datetime.now(UTC).isoformat(),
        microbatches=list(microbatches),
        batch_sizes=list(batch_sizes),
        arm_names=list(names),
        effective_source_sha256=code_hashes(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifests={},
        initialization_sha256={},
        rows=[],
        adopted_microbatch=8,
        scope="No corpus fits or automatic optimization adoption. Parity has dropout "
        "disabled; training timing uses normal dropout. Tolerances are engineering "
        "checks, not evidence of identical training trajectories.",
    )
    if device == "cuda" and report["runtime"]["cuda_device"]["total_memory_bytes"] >= 11 * 1024**3:
        raise ValueError("Pilot requires one 10GB MIG allocation")
    for recipe, objective in (("L4", "causal"), ("L2R2", "cbow")):
        root = parent / recipe
        manifest = load_manifest(root)
        report["manifests"][recipe] = manifest["fingerprint"]
        if manifest["settings"]["objectives"] != [objective]:
            raise ValueError("Unexpected parent objective")
        with np.load(root / f"{objective}.npz", allow_pickle=False) as data:
            ids = np.random.default_rng(619).choice(data["train_0"], max(batch_sizes), replace=True)
            contexts, targets = data["contexts"][ids], data["targets"][ids]
        save_npz(
            output / f"microbatch-inputs-{recipe}.npz", ids=ids, contexts=contexts, targets=targets
        )
        contexts, targets = torch.from_numpy(contexts), torch.from_numpy(targets)
        arms = {t["arm"]["name"]: t["arm"] for t in manifest["tasks"]}
        for name in names:
            arm = arms[name]
            torch.manual_seed(42)
            template = create_benchmark_model(
                manifest["vocabulary_size"], manifest["settings"]["window"], objective, arm
            )
            digest = hashlib.sha256()
            for key, tensor in template.state_dict().items():
                digest.update(key.encode() + b"\0" + tensor.detach().numpy().tobytes())
            report["initialization_sha256"][f"{recipe}/{name}"] = digest.hexdigest()

            for count in batch_sizes:
                reference = parity_state(
                    fresh_model(template, device),
                    contexts[:count],
                    targets[:count],
                    microbatches[0],
                    arm["lr"],
                )
                for micro in microbatches:
                    actual = parity_state(
                        fresh_model(template, device),
                        contexts[:count],
                        targets[:count],
                        micro,
                        arm["lr"],
                    )
                    checked = compare(reference, actual)
                    model = fresh_model(template, device)
                    measured = measure(
                        model, contexts[:count], targets[:count], micro, arm["lr"], repeats, device
                    )
                    report["rows"].append(
                        dict(
                            recipe=recipe,
                            objective=objective,
                            arm=name,
                            batch=count,
                            microbatch=micro,
                            parity=checked,
                            **measured,
                        )
                    )
                    del model
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    write_json(output / "microbatch-report.json", report)
    report["status"] = "complete"
    report["finished_at"] = datetime.now(UTC).isoformat()
    report["seconds"] = time.perf_counter() - begun
    report["all_parity_passed"] = all(r["parity"]["passed"] for r in report["rows"])
    write_json(output / "microbatch-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    report = profile(args.parent, args.output, args.device)
    print(
        f"Pilot complete: {len(report['rows'])} conditions; "
        f"all parity checks passed={report['all_parity_passed']}; training remains microbatch8"
    )


if __name__ == "__main__":
    main()
