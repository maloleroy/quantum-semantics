"""Serve a local page comparing next-token autocompletion from locked runs.

Optional pretrained references are shown for orientation only: they are trained on other
data with their own tokenizer, so they sit outside the controlled comparison.
"""

import argparse
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import parse_qs

import numpy as np
import torch
from torch import Tensor

from .data import Dataset, FloatArray, IntArray
from .generation import encode_prompt, generate
from .models import AdaptiveModel, WordModel, build_model, product_inputs, spectral_controls
from .quantum import orthonormal_columns, seed_random
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


def parameter_label(parameters: int) -> str:
    return f"{parameters:,}" if parameters < 1e6 else f"{parameters / 1e6:.3g}M"


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
    size = parameter_label(parameters)
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


class CausalOutput(Protocol):
    logits: Tensor


class ReferenceConfig(Protocol):
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    max_position_embeddings: int


class CausalModel(Protocol):
    config: ReferenceConfig

    def __call__(self, input_ids: Tensor) -> CausalOutput: ...
    def parameters(self) -> Iterator[Tensor]: ...
    def eval(self) -> object: ...


class TextCodec(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class Pretrained(Protocol):
    def from_pretrained(self, name: str, *, revision: str | None) -> object: ...


@dataclass(frozen=True)
class Reference:
    info: dict[str, object]
    model: CausalModel
    codec: TextCodec


def load_reference(label: str, source: str) -> Reference:
    """Load a Hugging Face causal language model, pinned as NAME@REVISION when given."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name, _, revision = source.partition("@")
    models, tokenizers = cast(Pretrained, AutoModelForCausalLM), cast(Pretrained, AutoTokenizer)
    model = cast(CausalModel, models.from_pretrained(name, revision=revision or None))
    codec = cast(TextCodec, tokenizers.from_pretrained(name, revision=revision or None))
    model = cast(CausalModel, cast(torch.nn.Module, model).float())
    model.eval()
    config = model.config
    parameters = sum(parameter.numel() for parameter in model.parameters())
    details = [
        ("comparison", "reference only, outside the controlled comparison"),
        ("training data", "its own pretraining corpus, not our WikiText split"),
        ("parameters", f"{parameters:,}"),
        ("tokenizer", f"its own, {config.vocab_size:,} output ids"),
        ("context", f"latest {config.max_position_embeddings:,} tokens"),
        (
            "model",
            f"{config.model_type}, {config.num_hidden_layers} layers, width {config.hidden_size}",
        ),
        ("checkpoint", f"{name} @ {revision[:7] or 'latest'}"),
    ]
    summary = f"pretrained reference · not comparable · {parameter_label(parameters)} parameters"
    info: dict[str, object] = {"name": label or name, "summary": summary, "details": details}
    return Reference(info, model, codec)


def complete_reference(
    reference: Reference, text: str, steps: int, argmax: bool
) -> dict[str, object]:
    """Same decoding rule and sampling seed as `generate`, with the model's own tokenizer.

    GPT-style tokenizers attach a space to the following word, so a prompt ending in spaces
    is out of distribution. Encode it without them and drop them from the continuation.
    """
    window = reference.model.config.max_position_embeddings
    prompt = text.rstrip(" ")
    spaces = " " * (len(text) - len(prompt))
    ids = reference.codec.encode(prompt)[-window:]
    if not ids:
        return {"text": "", "next": [], "ms": 0}
    started = time.perf_counter()
    seed_random(42)
    drawn: list[int] = []
    top: list[list[object]] = []
    with torch.no_grad():
        for _ in range(steps):
            context = torch.tensor([(ids + drawn)[-window:]])
            logits = reference.model(input_ids=context).logits[0, -1]
            probabilities = logits.double().softmax(-1)
            if not top:
                best = torch.topk(probabilities, 5)
                top = [
                    [reference.codec.decode([int(token)]), float(probability)]
                    for token, probability in zip(best.indices, best.values, strict=True)
                ]
            drawn.append(
                int(probabilities.argmax()) if argmax else int(torch.multinomial(probabilities, 1))
            )
    return {
        "text": reference.codec.decode(drawn).removeprefix(spaces),
        "next": top,
        "ms": round(1000 * (time.perf_counter() - started)),
    }


def handler(entries: list[Entry | Reference], steps: int) -> type[BaseHTTPRequestHandler]:
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
                entry, text = entries[int(index)], values.get("text", "")
                result = (
                    complete(entry, text, steps, mode == "argmax")
                    if isinstance(entry, Entry)
                    else complete_reference(entry, text, steps, mode == "argmax")
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
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        help="LABEL=NAME@REVISION: a pretrained Hugging Face model shown outside the comparison",
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
    entries: list[Entry | Reference] = []
    for argument in cast(list[str], args.runs):
        label, _, run = argument.rpartition("=")
        entries.append(load_entry(label, Path(run), cast(str, args.device)))
        print(f"Loaded {entries[-1].info['name']} from {run}", flush=True)
    for argument in cast(list[str], args.reference):
        label, _, source = argument.rpartition("=")
        entries.append(load_reference(label, source))
        print(f"Loaded {entries[-1].info['name']} from {source}", flush=True)
    server = HTTPServer(("127.0.0.1", port), handler(entries, steps))
    print(f"Serving http://127.0.0.1:{server.server_port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
