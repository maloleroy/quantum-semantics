"""Train a small causal semantic decoder: 10 epochs first, then resume to 50."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from qcse.data import DATASETS, build_vocabulary, load_corpus, make_examples, split_examples
from qcse.outputs import atomic_path, new_run_directory, run_status, write_json
from qcse.semantic import SemanticModel


def sanity_sentences():
    """96 distinct, repetitive templates; a trainability check, not an NLP benchmark."""
    return [
        f"the {color} {animal} eats {food} near the {place}".split()
        for color in ("red", "blue", "small", "large")
        for animal, food in (
            ("cat", "fish"),
            ("dog", "meat"),
            ("bird", "seeds"),
            ("horse", "grass"),
            ("rabbit", "carrots"),
            ("bear", "berries"),
        )
        for place in ("river", "house", "tree", "garden")
    ]


@torch.no_grad()
def evaluate(model, contexts, targets, ids, batch_size):
    loss = correct = top5 = cosine_correct = cosine_top5 = 0.0
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        scores = model.scores(contexts[batch])
        cosine_scores = model.scores(contexts[batch], similarity="cosine")
        target = targets[batch]
        loss += float(F.cross_entropy(scores, target, reduction="sum"))
        correct += float((scores.argmax(-1) == target).sum())
        top5 += float(
            (scores.topk(min(5, scores.shape[-1]), dim=-1).indices == target[:, None]).any(-1).sum()
        )
        cosine_correct += float((cosine_scores.argmax(-1) == target).sum())
        cosine_top5 += float(
            (cosine_scores.topk(min(5, scores.shape[-1]), dim=-1).indices == target[:, None])
            .any(-1)
            .sum()
        )
    ce = loss / len(ids)
    return dict(
        cross_entropy=ce,
        perplexity=math.exp(min(ce, 80)),
        top1=correct / len(ids),
        top5=top5 / len(ids),
        cosine_top1=cosine_correct / len(ids),
        cosine_top5=cosine_top5 / len(ids),
        examples=len(ids),
    )


def experiment(args, output, saved=None):
    torch.set_num_threads(args.threads)
    if saved is None:
        torch.manual_seed(args.seed)
        if args.datasets:
            corpus = load_corpus(
                datasets=args.datasets, max_sentences=args.max_sentences, seed=args.seed
            )
            sentences, vocabulary, provenance = corpus.sentences, corpus.vocabulary, corpus.summary
        else:
            sentences = sanity_sentences()
            vocabulary = build_vocabulary(sentences)
            provenance = {"dataset": "synthetic repetitive sanity corpus"}
        examples = make_examples(sentences, vocabulary, window=args.window, objective="causal")
        dev_ids, test_ids = split_examples(examples, sentences, args.seed)
        dev_examples = [examples[i] for i in dev_ids]
        train_local, val_local = split_examples(dev_examples, sentences, args.seed + 1)
        train_ids, val_ids = dev_ids[train_local], dev_ids[val_local]
        contexts = torch.full((len(examples), args.window), len(vocabulary), dtype=torch.long)
        for i, example in enumerate(examples):
            contexts[i, -len(example.context) :] = torch.tensor(example.context)
        targets = torch.tensor([example.target for example in examples])
        model = SemanticModel(
            len(vocabulary), args.window, args.embedding_dim, args.qubits, args.layers
        )
        settings = vars(args).copy() | {"output": str(output), "resume": None}
        rng = np.random.default_rng(args.seed)
        history = []
        with (output / "sentences.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sentence"])
            writer.writerows([[" ".join(words)] for words in sentences])
        write_json(output / "vocabulary.json", vocabulary)
        write_json(
            output / "summary.json",
            dict(
                provenance=provenance,
                settings=settings,
                model=model.config,
                parameters=sum(p.numel() for p in model.parameters()),
                topology="fixed latent chain: q -> q+1, controlled RZ",
                measurements="exact X/Y/Z expectations; zero sampling shots",
                backend="exact statevector; no MPS bond truncation",
                split="sentence-grouped 64/16/20 train/validation/test",
            ),
        )
    else:
        settings, vocabulary = saved["settings"], saved["vocabulary"]
        model = SemanticModel(**saved["model_config"])
        model.load_state_dict(saved["model"])
        contexts, targets = saved["contexts"], saved["targets"]
        train_ids, val_ids, test_ids = (saved[key] for key in ("train_ids", "val_ids", "test_ids"))
        rng = np.random.default_rng()
        rng.bit_generator.state = saved["rng_state"]
        history = saved["history"]
    if settings["ansatz"] == "zero":
        with torch.no_grad():
            model.ansatz.zero_()
    if settings["ansatz"] in ("frozen", "zero"):
        model.ansatz.requires_grad_(False)
    if settings["trainable"] == "decoder":
        model.set_trainable({"decoder"})
    model.to(args.device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=settings["learning_rate"],
    )
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
    contexts, targets = contexts.to(args.device), targets.to(args.device)
    train_ids, val_ids, test_ids = [np.asarray(ids) for ids in (train_ids, val_ids, test_ids)]
    # Fixed monitoring sample; training continues drawing from the complete train split.
    monitor = np.random.default_rng(settings["seed"] + 2)
    train_monitor = monitor.choice(train_ids, min(512, len(train_ids)), replace=False)
    gradients = {}

    def checkpoint():
        payload = dict(
            model_config=model.config,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            settings=settings,
            vocabulary=vocabulary,
            contexts=contexts.cpu(),
            targets=targets.cpu(),
            train_ids=train_ids.tolist(),
            val_ids=val_ids.tolist(),
            test_ids=test_ids.tolist(),
            rng_state=rng.bit_generator.state,
            history=history,
        )
        with atomic_path(output / "checkpoint.pt") as temporary:
            torch.save(payload, temporary)
        write_json(output / "history.json", history)

    def record(epoch, elapsed=0.0):
        train_metrics = evaluate(model, contexts, targets, train_monitor, args.batch_size)
        validation_metrics = evaluate(model, contexts, targets, val_ids, args.batch_size)
        row = dict(
            epoch=epoch,
            train=train_metrics,
            validation=validation_metrics,
            entanglement=model.entanglement(contexts[train_monitor[:32]]),
            gradient_norms=gradients.copy(),
            seconds=elapsed,
        )
        history.append(row)
        checkpoint()
        print(
            f"Epoch {epoch:3d}: train CE={train_metrics['cross_entropy']:.4f}, "
            f"val CE={validation_metrics['cross_entropy']:.4f}, "
            f"val top1={validation_metrics['top1']:.3f}, top5={validation_metrics['top5']:.3f}",
            flush=True,
        )

    if not history:
        record(0)
    start = history[-1]["epoch"]
    if args.epochs <= start:
        raise ValueError(f"Epoch target must exceed saved epoch {start}")
    for epoch in range(start + 1, args.epochs + 1):
        begun = time.monotonic()
        order = rng.choice(train_ids, settings["samples_per_epoch"], replace=True)
        for offset in range(0, len(order), settings["batch_size"]):
            batch = order[offset : offset + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model.scores(contexts[batch]), targets[batch])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            gradients = {
                name: float(parameter.grad.norm())
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            gradients["total_before_clip"] = float(norm)
            optimizer.step()
        record(epoch, time.monotonic() - begun)
    # Score test only at the requested final stage, after the 10-epoch diagnostic.
    if args.evaluate_test:
        result = evaluate(model, contexts, targets, test_ids, args.batch_size)
        counts = torch.bincount(targets[train_ids].cpu(), minlength=len(vocabulary)).float()
        frequent = counts.topk(min(5, len(vocabulary))).indices
        held = targets[test_ids].cpu()
        result["unigram_top1"] = float((held == frequent[0]).float().mean())
        result["unigram_top5"] = float((held[:, None] == frequent).any(-1).float().mean())
        result["epoch"] = args.epochs
        write_json(output / "test.json", result)
        words = dict(enumerate(vocabulary))
        words[len(vocabulary)] = "<pad>"
        with torch.no_grad():
            sample = contexts[test_ids[:8]]
            predicted = (
                model.scores(sample).topk(min(5, len(vocabulary)), dim=-1).indices.cpu().tolist()
            )
        write_json(
            output / "predictions.json",
            [
                dict(
                    context=[words[i] for i in context],
                    target=words[int(target)],
                    top5=[words[i] for i in guesses],
                )
                for context, target, guesses in zip(
                    sample.cpu().tolist(), held[:8], predicted, strict=True
                )
            ],
        )
        print(f"Held-out test: {json.dumps(result)}", flush=True)
    print(f"Saved: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", type=Path, help="Existing run directory; epochs is the total target"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/semantic"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        help="Omit for the repetitive sanity corpus",
    )
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--ansatz", choices=("trainable", "frozen", "zero"), default="trainable")
    parser.add_argument("--trainable", choices=("all", "decoder"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Score held-out test after the final training stage",
    )
    args = parser.parse_args()
    if min(args.epochs, args.samples_per_epoch, args.batch_size, args.threads) < 1:
        parser.error("epochs, samples, batch size and threads must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be positive and finite")
    runtime = dict(
        device=args.device, torch_version=torch.__version__, prototype="semantic decoder"
    )
    if args.resume:
        saved = torch.load(args.resume / "checkpoint.pt", map_location="cpu", weights_only=True)
        with run_status(args.resume, runtime):
            experiment(args, args.resume, saved)
    else:
        with new_run_directory(args.output, "semantic-causal", runtime) as output:
            experiment(args, output)


if __name__ == "__main__":
    main()
