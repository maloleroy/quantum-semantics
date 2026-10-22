"""Bounded CPU equivalence/timing check for the analytic product-state readout."""

import argparse
import copy
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from qcse.benchmark_models import HybridCircuitModel
from qcse.semantic import SemanticModel


def measure(repeats=5):
    torch.set_num_threads(1)
    rows = []
    for qubits in (2, 4, 8):
        torch.manual_seed(619)
        model = HybridCircuitModel(64, 4, dict(qubits=qubits, ansatz="identity")).double()
        reference = copy.deepcopy(model)
        contexts = torch.randint(0, 64, (8, 4))
        contexts[0, :2] = 64
        targets = torch.randint(0, 64, (8,))

        def vector(reference=reference, contexts=contexts):
            hidden = SemanticModel.representation(reference, contexts)
            return F.linear(hidden, reference.embedding.weight[:-1], reference.output_bias)

        left, right = model(contexts), vector()
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
        F.cross_entropy(left, targets).backward()
        F.cross_entropy(right, targets).backward()
        gradient_error = 0.0
        for (name, a), (other, b) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert name == other
            if a.grad is not None:
                assert b.grad is not None
                torch.testing.assert_close(a.grad, b.grad, atol=1e-12, rtol=1e-12)
                gradient_error = max(gradient_error, float((a.grad - b.grad).abs().max()))
        timings = {}
        for name, instance, forward in (
            ("analytic", model, lambda model=model, contexts=contexts: model(contexts)),
            ("statevector", reference, vector),
        ):
            samples = []
            for iteration in range(repeats + 2):
                instance.zero_grad(set_to_none=True)
                start = time.perf_counter()
                F.cross_entropy(forward(), targets).backward()
                seconds = time.perf_counter() - start
                if iteration >= 2:
                    samples.append(seconds)
            timings[name] = statistics.median(samples)
        rows.append(
            dict(
                qubits=qubits,
                max_logit_error=float((left - right).detach().abs().max()),
                max_gradient_error=gradient_error,
                median_seconds=timings,
                cpu_speed_ratio=timings["statevector"] / timings["analytic"],
            )
        )
    return dict(
        platform=platform.platform(),
        torch=torch.__version__,
        threads=1,
        dtype="float64",
        batch=8,
        vocabulary=64,
        repeats=repeats,
        measurements=rows,
        scope="Small CPU forward/backward microbenchmark, not full training or DGX speedup.",
        source_sha256={
            name: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / name).read_bytes()
            ).hexdigest()
            for name in (
                "src/qcse/benchmark_models.py",
                "src/qcse/semantic.py",
                "scripts/profile_identity_readout.py",
            )
        },
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    result = measure()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
