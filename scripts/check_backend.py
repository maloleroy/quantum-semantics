"""Fail-fast device check against Qiskit at the full-corpus qubit/layer scale."""

import argparse
import json
import os
import sys

import numpy as np
import torch
from qiskit.quantum_info import Statevector

from qcse.model import QCSEModel


def check_backend(device, qubits=14, layers=64):
    diagnostics = {
        "device": device,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "qubits": qubits,
        "layers": layers,
    }
    print(json.dumps(diagnostics, indent=2), flush=True)
    model = QCSEModel(
        [str(i) for i in range(2**qubits)], layers=layers, device=device, simulation_batch_size=2
    )
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        print(
            f"GPU: {properties.name}; visible memory: {properties.total_memory / 2**30:.2f} GiB",
            flush=True,
        )
    rng = np.random.default_rng(19)
    amplitudes = rng.normal(size=(3, 2**qubits)) + 1j * rng.normal(size=(3, 2**qubits))
    amplitudes /= np.linalg.norm(amplitudes, axis=1, keepdims=True)
    states = [Statevector(row) for row in amplitudes]
    weights = np.stack([model.weights, model.weights + 0.1])
    actual = model.predict_encoded(states, weights)
    bits = model.simulator.bits.cpu().numpy()
    expected = np.array(
        [
            [state.evolve(model.bound_ansatz(values)).probabilities() @ bits for state in states]
            for values in weights
        ]
    )
    tolerance = 1e-12 if device == "cpu" else 2e-5
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    model.simulator.batch_size = 1
    sequential = np.stack([model.predict_encoded(states, values) for values in weights])
    np.testing.assert_allclose(actual, sequential, atol=tolerance, rtol=tolerance)
    result = diagnostics | {
        "max_absolute_error_vs_qiskit": float(np.max(np.abs(actual - expected))),
        "max_absolute_error_vs_sequential": float(np.max(np.abs(actual - sequential))),
        "status": "passed",
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    args = parser.parse_args()
    try:
        check_backend(args.device)
    except (ValueError, RuntimeError, AssertionError) as error:
        print(f"Backend check failed: {error}", file=sys.stderr)
        if args.device == "cuda":
            print(
                "Run inside a GPU allocation using the `uv sync --locked` environment. "
                "Check nvidia-smi, torch.version.cuda and CUDA_VISIBLE_DEVICES; "
                "do not replace Slurm's device mask.",
                file=sys.stderr,
            )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
