"""Measure initialization-gradient variance versus register size, depth and cost locality."""

import argparse
import hashlib
import os
from pathlib import Path

import torch

from qcse.benchmark import code_hashes, runtime, save_code_snapshot
from qcse.outputs import new_run_directory, save_npz, write_json
from qcse.trainability import probe_gradients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/trainability"))
    parser.add_argument("--qubits", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--init-scales", type=float, nargs="+", default=[0.05, 1.0])
    parser.add_argument("--seeds", type=int, default=32, help="Initialization seeds starting at 42")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if (
        args.seeds < 2
        or min(args.batch_size, args.threads, *args.layers) < 1
        or not all(2 <= q <= 16 for q in args.qubits)
        or not all(0 < scale < float("inf") for scale in args.init_scales)
    ):
        parser.error("Invalid register/depth/initialization/sample budget")
    if any(
        len(values) != len(set(values)) for values in (args.qubits, args.layers, args.init_scales)
    ):
        parser.error("Grid values must be distinct")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    info = runtime(args.device)
    info.update(settings={k: v for k, v in vars(args).items() if k != "output"})
    info["code_sha256"] = code_hashes()
    info["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows = []
    with new_run_directory(args.output, "gradient-probe", info) as output:
        save_code_snapshot(output, info["code_sha256"])
        for qubits in args.qubits:
            for layers in args.layers:
                for scale in args.init_scales:
                    summary, arrays = probe_gradients(
                        qubits,
                        layers,
                        scale,
                        list(range(42, 42 + args.seeds)),
                        args.batch_size,
                        args.device,
                    )
                    save_npz(output / f"q{qubits}-depth{layers}-init{scale:g}.npz", **arrays)
                    rows.extend(summary)
                    write_json(
                        output / "summary.json",
                        dict(
                            rows=rows,
                            scope="Noiseless initial-gradient probe on fixed synthetic inputs; "
                            "Local/global observable controls, "
                            "not an NLP loss or plateau diagnosis. "
                            "Known final diagonal zeros excluded; causal-cone zeros can remain.",
                        ),
                    )
                    print(f"Measured q={qubits}, depth={layers}, init std={scale:g}", flush=True)
        print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
