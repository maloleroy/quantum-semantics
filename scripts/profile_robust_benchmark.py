"""Measure actual adapter batch costs; extrapolate exposure/evaluation budgets."""

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from qcse.benchmark import code_hashes, evaluate, training_loss
from qcse.benchmark_models import create_benchmark_model
from qcse.outputs import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-audit/profile.json"))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--objective", choices=("causal", "cbow"), default="causal")
    args = parser.parse_args()
    if min(args.repeats, args.threads) < 1:
        parser.error("Repeats and threads must be positive")
    torch.set_num_threads(args.threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((args.study / "manifest.json").read_text())
    settings = manifest["settings"]
    with np.load(args.study / f"{args.objective}.npz") as data:
        ids = np.random.default_rng(619).choice(data["train_0"], 64, replace=False)
        contexts = torch.as_tensor(data["contexts"][ids])
        targets = torch.as_tensor(data["targets"][ids])

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()
        elif args.device == "mps":
            torch.mps.synchronize()

    rows = []
    output = dict(
        device=args.device,
        platform=platform.platform(),
        torch=torch.__version__,
        threads=args.threads,
        batch=64,
        microbatch=settings["microbatch_size"],
        repeats=args.repeats,
        manifest=manifest["fingerprint"],
        objective=args.objective,
        effective_source_sha256=code_hashes(),
        measurements=rows,
    )
    arms = []
    for task in manifest["tasks"]:
        if task["objective"] == args.objective and task["arm"] not in arms:
            arms.append(task["arm"])
    if not arms:
        parser.error("The selected objective has no prepared tasks")
    for arm in arms:
        if args.arms and arm["name"] not in args.arms:
            continue
        torch.manual_seed(42)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = create_benchmark_model(
            manifest["vocabulary_size"], settings["window"], args.objective, arm
        ).to(args.device)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=arm["lr"]) if parameters else None
        training, inference = [], []
        for repetition in range(args.repeats + 1):
            model.train()
            sync()
            start = time.perf_counter()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, 64, settings["microbatch_size"]):
                    part = slice(offset, offset + settings["microbatch_size"])
                    loss = training_loss(
                        model, contexts[part].to(args.device), targets[part].to(args.device)
                    )
                    (loss * len(contexts[part]) / 64).backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
                optimizer.step()
            sync()
            trained = time.perf_counter() - start
            start = time.perf_counter()
            evaluate(model, contexts, targets, np.arange(64), settings["microbatch_size"])
            sync()
            evaluated = time.perf_counter() - start
            if repetition:
                training.append(trained)
                inference.append(evaluated)
        row = dict(
            arm=arm,
            training_batch_seconds=float(np.median(training)),
            evaluation_batch_seconds=float(np.median(inference)),
            train_measurements=training,
            eval_measurements=inference,
            parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in parameters),
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            cuda_peak_allocated=torch.cuda.max_memory_allocated()
            if args.device == "cuda"
            else None,
            cuda_peak_reserved=torch.cuda.max_memory_reserved() if args.device == "cuda" else None,
        )
        rows.append(row)
        write_json(args.output, output)
        print(json.dumps(row), flush=True)
        if optimizer is not None:
            del loss
        del model, optimizer, parameters
        if args.device == "cuda":
            torch.cuda.empty_cache()
    print(args.output)


if __name__ == "__main__":
    main()
