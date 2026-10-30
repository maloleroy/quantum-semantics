"""Autoregressive simulator draws without temperature, rejection or word decoding heads."""

from dataclasses import dataclass

import torch

from .data import Dataset, tokenize
from .models import WordModel
from .quantum import seed_random


def encode_prompt(dataset: Dataset, text: str) -> list[int]:
    if dataset.byte_pairs is not None:
        return dataset.byte_pairs.encode(text)
    lookup = {word: index for index, word in enumerate(dataset.vocabulary)}
    return [lookup.get(word, 0) for word in tokenize(text)]


@dataclass(frozen=True)
class Generation:
    text: str
    tokens: tuple[int, ...]
    non_text_outcomes: int


def generate(
    dataset: Dataset,
    model: WordModel,
    text: str,
    steps: int = 64,
    seed: int = 42,
    argmax: bool = False,
) -> Generation:
    ids = encode_prompt(dataset, text)
    if not ids or steps < 1:
        raise ValueError("Supply a nonempty prompt and positive generation length")
    seed_random(seed)
    drawn: list[int] = []
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            history = ids[-dataset.window :]
            context = torch.full((1, dataset.window), -1, dtype=torch.int64)
            context[0, -len(history) :] = torch.tensor(history)
            probabilities = model(context)[0]
            selected = (
                int(probabilities.argmax()) if argmax else int(torch.multinomial(probabilities, 1))
            )
            ids.append(selected)
            drawn.append(selected)
    if dataset.byte_pairs is None:
        return Generation(
            " ".join(dataset.vocabulary[token] for token in ids), tuple(drawn), drawn.count(0)
        )
    pieces = dataset.byte_pairs.pieces
    # Preserve every outcome. Invalid text outcomes stay explicit, including in future contexts.
    fragments = [pieces[token] or dataset.vocabulary[token].encode("utf-8") for token in drawn]
    rendered = (text.encode("utf-8") + b"".join(fragments)).decode("utf-8", errors="replace")
    return Generation(rendered, tuple(drawn), sum(not pieces[token] for token in drawn))
