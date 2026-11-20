"""Serve a local page comparing next-token autocompletion from locked runs."""

import argparse
import json
import os
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs

import numpy as np
import torch
from torch import Tensor

from .data import Dataset, FloatArray, IntArray
from .generation import encode_prompt, generate
from .models import AdaptiveModel, WordModel, build_model, product_inputs, spectral_controls
from .quantum import orthonormal_columns
from .storage import fingerprint, load_weights, read_json
from .training import FitRecord, model_config

PAGE = Path(__file__).with_name("demo.html")


class CachedBases(WordModel):
    """Inference-only AdaptiveModel forward that orthonormalizes each factor once.

    The basis depends only on weights, while energies still shift with context. A QR of
    the 65,536 x 128 factors costs seconds on a laptop CPU and dominates every token.
    Row-wise matrix-vector products are about twice as fast as 1 x N matrix products.
    """

    def __init__(self, model: AdaptiveModel) -> None:
        super().__init__(model.config)
        self.inner = model
        with torch.no_grad():
            bases = [orthonormal_columns(factors) for factors in model.factors]
        self.bases = [(basis.conj().T.contiguous(), basis.contiguous()) for basis in bases]

    def forward(self, contexts: Tensor) -> Tensor:
        model = self.inner
        states, controls = product_inputs(
            contexts, model.token_angles, model.control_offsets, model.config
        )
        shifts = spectral_controls(contexts, model.spectral_angles, model.config)
        for layer, (adjoint, basis) in enumerate(self.bases):
            phases = torch.exp(-1j * (model.energies[layer] + shifts)) - 1
            states = states + torch.stack(
                [
                    torch.mv(basis, torch.mv(adjoint, state) * phase)
                    for state, phase in zip(states, phases, strict=True)
                ]
            )
            if layer < len(self.bases) - 1:
                states = states * torch.exp(-0.5j * (controls[:, layer] @ model.basis_z.T))
        return model.word_probabilities(states)


@dataclass(frozen=True)
class Entry:
    info: dict[str, object]
    dataset: Dataset
    model: WordModel
    pieces: tuple[bytes, ...]


def check_validation(model: WordModel, dataset: Dataset, saved: Path, rows: int) -> None:
    """Weights trained under older source hashes must still reproduce their saved scores."""
    with np.load(saved, allow_pickle=False) as arrays:
        expected = cast(FloatArray, arrays["success"])[:rows]
        targets = cast(IntArray, arrays["targets"])[:rows]
    if not np.array_equal(targets, dataset.validation.targets[: len(targets)]):
        raise ValueError("Saved validation targets do not match the run dataset")
    contexts = torch.tensor(dataset.validation.contexts[: len(targets)], device=model.device)
    with torch.no_grad():
        probabilities = model(contexts)
    picked = torch.arange(len(targets), device=model.device)
    observed = probabilities[picked, torch.tensor(targets, device=model.device)].cpu().numpy()
    if not np.allclose(observed, expected, rtol=1e-6, atol=1e-12):
        raise ValueError("Locked weights do not reproduce saved validation probabilities")


def describe(label: str, record: FitRecord, epochs: object) -> dict[str, object]:
    config = model_config(record["model"])
    protocol = record["protocol"]
    metrics = cast(dict[str, float | None], record["validation"])
    per_byte = metrics["target_bits_per_byte"]
    classical, spectral = protocol.startswith("classical"), "auxiliary" in protocol
    qubits = config.size.bit_length() - 1
    if spectral:
        qubits += config.ancilla_size.bit_length() - 1
    model = config.kind
    if not classical:
        model += f", {config.layers} layers" + (f", rank {config.rank}" if spectral else "")
    context = config.window if config.qcse_window is None else config.qcse_window
    parameters = record["trainable_parameters"]
    size = f"{parameters:,}" if parameters < 1e6 else f"{parameters / 1e6:.3g}M"
    details = [
        ("validation NLL", f"{metrics['nll']:.3f} nats"),
        ("bits per byte", "—" if per_byte is None else f"{per_byte:.3f}"),
        ("sampling success", f"{metrics['success']:.2%}"),
        ("argmax accuracy", f"{metrics['accuracy']:.2%}"),
        ("parameters", f"{parameters:,}"),
        ("context", f"latest {context} tokens"),
        ("readout", protocol),
        ("model", model),
        ("checkpoint", f"seed {record['seed']}, epoch {record['selected_epoch']} of {epochs}"),
    ]
    return {
        "name": label,
        "summary": " · ".join(
            ("classical" if classical else f"{qubits} qubits", f"{config.size:,} tokens", size)
        )
        + " parameters",
        "details": details,
    }


def load_entry(label: str, run: Path, device: str, rows: int = 8) -> Entry:
    selection = read_json(run / "selection.json")
    records = cast(list[FitRecord], selection["fits"])
    if len(records) != 1:
        raise ValueError(f"{run} must hold exactly one locked fit")
    record = records[0]
    weights = run / record["name"] / "weights.npz"
    if fingerprint(run / "dataset.npz") != selection["dataset_sha256"]:
        raise ValueError("Dataset changed after selection")
    if fingerprint(weights) != record["weights_sha256"]:
        raise ValueError("Weights changed after selection")
    dataset = Dataset.load(run / "dataset.npz")
    model = build_model(model_config(record["model"]), dataset, record["seed"], device)
    load_weights(weights, model)
    model.eval()
    if isinstance(model, AdaptiveModel):
        model = CachedBases(model)
    check_validation(model, dataset, run / record["name"] / "validation.npz", rows)
    codec = dataset.byte_pairs
    pieces = (
        tuple((" " + word).encode() for word in dataset.vocabulary)
        if codec is None
        else tuple(
            piece or word.encode()
            for piece, word in zip(codec.pieces, dataset.vocabulary, strict=True)
        )
    )
    epochs = cast(dict[str, object], selection["training"])["epochs"]
    return Entry(describe(label or record["name"], record, epochs), dataset, model, pieces)


def complete(entry: Entry, text: str, steps: int, argmax: bool) -> dict[str, object]:
    """Draw a continuation exactly as `qcse generate`, plus the first-step distribution."""
    ids = encode_prompt(entry.dataset, text)[-entry.dataset.window :]
    if not ids:
        return {"text": "", "next": [], "ms": 0}
    started = time.perf_counter()
    context = torch.full((1, entry.dataset.window), -1, dtype=torch.int64)
    context[0, -len(ids) :] = torch.tensor(ids)
    with torch.no_grad():
        probabilities = entry.model(context.to(entry.model.device))[0]
    top = torch.topk(probabilities, 5)
    tokens, values = [int(token) for token in top.indices], [float(p) for p in top.values]
    drawn = generate(entry.dataset, entry.model, text, steps, argmax=argmax).tokens
    render = b"".join(entry.pieces[token] for token in drawn).decode("utf-8", errors="replace")
    return {
        "text": render,
        "next": [
            [entry.pieces[token].decode("utf-8", errors="replace"), probability]
            for token, probability in zip(tokens, values, strict=True)
        ],
        "ms": round(1000 * (time.perf_counter() - started)),
    }


def handler(entries: list[Entry], steps: int) -> type[BaseHTTPRequestHandler]:
    page = PAGE.read_bytes()
    models = json.dumps([entry.info for entry in entries]).encode()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, body: bytes, kind: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path, _, query = self.path.partition("?")
            values = {
                key: items[0] for key, items in parse_qs(query, keep_blank_values=True).items()
            }
            index, mode = values.get("model", ""), values.get("mode", "argmax")
            if path == "/":
                self.reply(page, "text/html; charset=utf-8")
            elif path == "/models":
                self.reply(models, "application/json")
            elif (
                path == "/complete"
                and index.isdigit()
                and int(index) < len(entries)
                and mode in {"argmax", "sample"}
            ):
                result = complete(
                    entries[int(index)], values.get("text", ""), steps, mode == "argmax"
                )
                self.reply(json.dumps(result).encode(), "application/json")
            else:
                self.send_error(404)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", nargs="+", help="LABEL=RUN or RUN; each run directory holds one locked fit"
    )
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=os.environ.get("QCSE_DEVICE", "cpu")
    )
    args = parser.parse_args()
    steps, port, threads = cast(int, args.steps), cast(int, args.port), cast(int, args.threads)
    if steps < 1 or threads < 1:
        parser.error("Steps and threads must be positive")
    torch.set_num_threads(threads)
    entries: list[Entry] = []
    for argument in cast(list[str], args.runs):
        label, _, run = argument.rpartition("=")
        entries.append(load_entry(label, Path(run), cast(str, args.device)))
        print(f"Loaded {entries[-1].info['name']} from {run}", flush=True)
    server = HTTPServer(("127.0.0.1", port), handler(entries, steps))
    print(f"Serving http://127.0.0.1:{server.server_port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
