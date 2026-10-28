"""Prepare a corpus, fit models, lock/evaluate a run, or sample the next word."""

import argparse
from pathlib import Path
from typing import Literal

import torch

from .data import Dataset, prepare, tokenize
from .encoding import ContextMethod
from .metrics import plot_history
from .models import MODEL_KINDS, ModelConfig, ModelKind
from .quantum import seed_random
from .storage import new_directory
from .training import TrainConfig, load_predictor, test, train


class Arguments(argparse.Namespace):
    command: str
    data: Path
    output: Path
    dataset: Path
    history: Path
    run: Path
    fit: str
    text: str
    vocabulary: int
    window: int
    train_limit: int
    validation_limit: int
    test_limit: int
    split_seed: int
    models: list[ModelKind]
    seeds: list[int]
    epochs: int
    batch_size: int
    lr: float
    threads: int
    layers: int
    rank: int
    phase_scale: float
    latent: int
    encoder: Literal["auto", "amplitude", "qcse"]
    context_method: ContextMethod
    decoder: Literal["born", "rotated", "marginal"]
    freeze_core: bool
    selection: Literal["success", "nll"]
    loss: Literal["success", "nll"]
    argmax: bool
    seed: int


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser(
        "prepare",
        help="Prepare sentence groups or a WikiText directory with official article splits",
    )
    data.add_argument(
        "--data",
        type=Path,
        default=Path("data/phrases.csv"),
        help="Sentence CSV/text file, or directory containing wiki.{train,valid,test}.raw",
    )
    data.add_argument("--output", type=Path, default=Path("output"))
    data.add_argument("--vocabulary", type=int, default=64)
    data.add_argument("--window", type=int, default=4)
    data.add_argument("--train-limit", type=int, default=32768)
    data.add_argument("--validation-limit", type=int, default=4096)
    data.add_argument("--test-limit", type=int, default=8192)
    data.add_argument("--split-seed", type=int, default=20261026)
    fitting = commands.add_parser("train", help="Validation selection only; never score test")
    fitting.add_argument("--dataset", type=Path, required=True)
    fitting.add_argument("--output", type=Path, default=Path("output"))
    fitting.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KINDS,
        default=["hamiltonian", "dephased", "mlp", "bigram"],
    )
    fitting.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    fitting.add_argument("--epochs", type=int, default=30)
    fitting.add_argument("--batch-size", type=int, default=128)
    fitting.add_argument("--lr", type=float, default=0.001)
    fitting.add_argument("--rank", type=int, default=32)
    fitting.add_argument("--phase-scale", type=float, default=1.0)
    fitting.add_argument("--layers", type=int, default=2)
    fitting.add_argument("--latent", type=int, default=16)
    fitting.add_argument("--encoder", choices=["auto", "amplitude", "qcse"], default="auto")
    fitting.add_argument(
        "--context-method",
        choices=["exponential", "diagonal", "phase", "hash", "angular"],
        default="exponential",
    )
    fitting.add_argument("--decoder", choices=["born", "rotated", "marginal"], default="born")
    fitting.add_argument("--freeze-core", action="store_true")
    fitting.add_argument("--selection", choices=["success", "nll"], default="success")
    fitting.add_argument("--loss", choices=["nll", "success"], default="nll")
    scoring = commands.add_parser("evaluate", help="Score every locked fit once on held-out test")
    scoring.add_argument("--run", type=Path, required=True)
    predicting = commands.add_parser(
        "predict", help="Sample a next word; argmax is a separate diagnostic"
    )
    predicting.add_argument("--run", type=Path, required=True)
    predicting.add_argument("--fit", required=True)
    predicting.add_argument("--text", required=True)
    predicting.add_argument("--seed", type=int, default=42)
    predicting.add_argument("--argmax", action="store_true")
    plotting = commands.add_parser("plot", help="Plot sampling, argmax and loss from a fit history")
    plotting.add_argument("--history", type=Path, required=True)
    for command in (fitting, scoring, predicting):
        command.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(namespace=Arguments())
    if args.command not in {"prepare", "plot"} and args.threads < 1:
        parser.error("Threads must be positive")
    return args


def main() -> None:
    args = parse_arguments()
    if args.command == "plot":
        print(plot_history(args.history))
        return
    if args.command == "prepare":
        output = new_directory(args.output, "dataset")
        dataset = prepare(
            args.data,
            output,
            args.vocabulary,
            args.window,
            args.train_limit,
            args.validation_limit,
            args.test_limit,
            args.split_seed,
        )
        print(f"Dataset: {output / 'dataset.npz'} ({dataset.size} words)")
        return
    torch.set_num_threads(args.threads)
    if args.command == "train":
        dataset = Dataset.load(args.dataset)
        configs = [
            ModelConfig(
                kind,
                dataset.size,
                dataset.window,
                layers=args.layers,
                rank=args.rank,
                phase_scale=args.phase_scale,
                encoder=args.encoder,
                context_method=args.context_method,
                decoder=args.decoder,
                latent=args.latent,
                freeze_core=args.freeze_core,
            )
            for kind in args.models
        ]
        settings = TrainConfig(args.epochs, args.batch_size, args.lr, args.selection, args.loss)
        train(args.dataset, configs, settings, args.seeds, args.output)
    elif args.command == "evaluate":
        print(test(args.run))
    else:
        dataset, model = load_predictor(args.run, args.fit)
        words = tokenize(args.text)
        if not words:
            raise ValueError("Supply a nonempty token context")
        lookup = {word: index for index, word in enumerate(dataset.vocabulary)}
        ids = [lookup.get(word, 0) for word in words][-dataset.window :]
        context = torch.full((1, dataset.window), -1, dtype=torch.int64)
        context[0, -len(ids) :] = torch.tensor(ids)
        seed_random(args.seed)
        with torch.no_grad():
            probabilities = model(context)[0]
            selected = (
                int(probabilities.argmax())
                if args.argmax
                else int(torch.multinomial(probabilities, 1))
            )
        print(dataset.vocabulary[selected])
        print("argmax diagnostic" if args.argmax else f"ideal simulator sample: {model.protocol}")


if __name__ == "__main__":
    main()
