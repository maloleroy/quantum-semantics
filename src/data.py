"""Explicit corpora, train-only vocabulary/statistics and duplicate-safe causal splits."""

import csv
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from .storage import Array, fingerprint, save_arrays, write_json
from .tokenization import BytePairs

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
HEADERS = {"sentence", "sentences", "cleaned_sentence", "text", "word", "words"}
WIKITEXT_FILES = {
    "train": "wiki.train.raw",
    "validation": "wiki.valid.raw",
    "test": "wiki.test.raw",
}
ARTICLE_TITLE = re.compile(r"^= [^=].* =$", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(unicodedata.normalize("NFKC", text).lower().replace("’", "'"))


def load_sentences(path: Path) -> list[list[str]]:
    sentences: list[list[str]] = []
    first = True
    with path.open(encoding="utf-8-sig", newline="") as source:
        for line in source:
            text = line.strip()
            if not text:
                continue
            if text.startswith('"') and text.endswith('"'):
                fields = next(csv.reader([text]))
                if len(fields) == 1:
                    text = fields[0]
            if first and text.lower() in HEADERS:
                first = False
                continue
            first = False
            words = tokenize(text)
            if words:
                sentences.append(words)
    if not sentences:
        raise ValueError(f"No sentences in {path}")
    return sentences


def load_wikitext(path: Path) -> dict[str, list[list[str]]]:
    """Keep official partitions and allow causal context across an article's lines."""
    partitions: dict[str, list[list[str]]] = {}
    for name, filename in WIKITEXT_FILES.items():
        articles: list[list[str]] = []
        current: list[str] = []
        with (path / filename).open(encoding="utf-8") as source:
            for line in source:
                text = line.strip()
                if not text:
                    continue
                if ARTICLE_TITLE.fullmatch(text):
                    if current:
                        articles.append(current)
                    current = tokenize(text)
                else:
                    if not current:
                        raise ValueError(f"Expected an article title before prose in {filename}")
                    current.extend(tokenize(text))
        if current:
            articles.append(current)
        if not articles:
            raise ValueError(f"No WikiText articles in {filename}")
        partitions[name] = articles
    return partitions


@dataclass(frozen=True)
class Partition:
    contexts: IntArray
    targets: IntArray
    groups: IntArray
    target_bytes: IntArray | None = None


@dataclass(frozen=True)
class Dataset:
    vocabulary: tuple[str, ...]
    train: Partition
    validation: Partition
    test: Partition
    transitions: FloatArray
    unigram: FloatArray
    byte_pairs: BytePairs | None = None

    @property
    def size(self) -> int:
        return len(self.vocabulary)

    @property
    def window(self) -> int:
        return self.train.contexts.shape[1]

    def save(self, path: Path) -> None:
        arrays: dict[str, Array] = {
            "vocabulary": np.asarray(self.vocabulary, dtype=np.str_),
            "transitions": self.transitions,
            "unigram": self.unigram,
        }
        if self.byte_pairs is not None:
            arrays["byte_merges"] = np.asarray(self.byte_pairs.merges, dtype=np.int64).reshape(
                -1, 2
            )
        for name, part in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            arrays.update(
                {
                    f"{name}_contexts": part.contexts,
                    f"{name}_targets": part.targets,
                    f"{name}_groups": part.groups,
                }
            )
            if part.target_bytes is not None:
                arrays[f"{name}_target_bytes"] = part.target_bytes
        save_arrays(path, arrays)

    @staticmethod
    def load(path: Path) -> "Dataset":
        with np.load(path, allow_pickle=False) as arrays:
            parts = [
                Partition(
                    np.asarray(arrays[f"{name}_contexts"], dtype=np.int64),
                    np.asarray(arrays[f"{name}_targets"], dtype=np.int64),
                    np.asarray(arrays[f"{name}_groups"], dtype=np.int64),
                    np.asarray(arrays[f"{name}_target_bytes"], dtype=np.int64)
                    if f"{name}_target_bytes" in arrays.files
                    else None,
                )
                for name in ("train", "validation", "test")
            ]
            vocabulary = tuple(
                str(word) for word in cast(npt.NDArray[np.str_], arrays["vocabulary"])
            )
            codec = None
            if "byte_merges" in arrays.files:
                merges = np.asarray(arrays["byte_merges"], dtype=np.int64)
                if merges.ndim != 2 or merges.shape[1] != 2:
                    raise ValueError("Expected two-column byte merge table")
                codec = BytePairs(len(vocabulary), tuple((int(a), int(b)) for a, b in merges))
                if vocabulary != codec.vocabulary:
                    raise ValueError("Byte vocabulary does not match the stored merge rules")
            dataset = Dataset(
                vocabulary,
                parts[0],
                parts[1],
                parts[2],
                np.asarray(arrays["transitions"], dtype=np.float64),
                np.asarray(arrays["unigram"], dtype=np.float64),
                codec,
            )
        if any(
            set(a.groups.tolist()) & set(b.groups.tolist())
            for a, b in (
                (dataset.train, dataset.validation),
                (dataset.train, dataset.test),
                (dataset.validation, dataset.test),
            )
        ):
            raise ValueError("Sentence groups overlap across partitions")
        if dataset.size < 2 or dataset.size & (dataset.size - 1):
            raise ValueError("The word register needs a power-of-two vocabulary")
        for part in (dataset.train, dataset.validation, dataset.test):
            if part.target_bytes is not None:
                if part.target_bytes.shape != part.targets.shape or np.any(part.target_bytes <= 0):
                    raise ValueError("Target byte lengths must be positive and match targets")
            if codec is not None:
                lengths = np.asarray([len(piece) for piece in codec.pieces], dtype=np.int64)
                if part.target_bytes is None or not np.array_equal(
                    part.target_bytes, lengths[part.targets]
                ):
                    raise ValueError("Target byte lengths disagree with byte vocabulary")
        return dataset


def bigram_probabilities(
    transitions: FloatArray, unigram: FloatArray, strength: float = 10.0
) -> FloatArray:
    if strength <= 0:
        raise ValueError("Smoothing strength must be positive")
    return (transitions + strength * unigram) / (transitions.sum(1, keepdims=True) + strength)


def prepare(
    source: Path,
    output: Path,
    vocabulary: int = 256,
    window: int = 4,
    train_limit: int = 32768,
    validation_limit: int = 4096,
    test_limit: int = 8192,
    seed: int = 20261026,
) -> Dataset:
    if vocabulary < 2 or vocabulary & (vocabulary - 1) or window < 1:
        raise ValueError("Use a power-of-two vocabulary >=2 and positive window")
    if min(train_limit, validation_limit, test_limit) < 0:
        raise ValueError("Limits must be nonnegative; zero means all eligible targets")
    official = load_wikitext(source) if source.is_dir() else None
    sentences = (
        [article for articles in official.values() for article in articles]
        if official is not None
        else load_sentences(source)
    )
    groups = sorted(set(map(tuple, sentences)))
    names = ("train", "validation", "test")
    membership: dict[tuple[str, ...], str] = {}
    if official is None:
        order = np.random.default_rng(seed).permutation(len(groups))
        n_test, n_validation = round(0.2 * len(groups)), round(0.16 * len(groups))
        membership = {
            groups[int(index)]: (
                "test"
                if position < n_test
                else "validation"
                if position < n_test + n_validation
                else "train"
            )
            for position, index in enumerate(order)
        }
    else:
        for name, articles in official.items():
            for article in articles:
                group = tuple(article)
                if group in membership and membership[group] != name:
                    raise ValueError("Normalized article overlaps across official partitions")
                membership[group] = name
    gids = {sentence: index for index, sentence in enumerate(groups)}
    counts = Counter(
        word
        for sentence in sentences
        if membership[tuple(sentence)] == "train"
        for word in sentence
    )
    size = 2 ** math.floor(math.log2(min(vocabulary, len(counts) + 1)))
    if size < 2:
        raise ValueError("Too few training words")
    words = ("<unk>", *sorted(counts, key=lambda word: (-counts[word], word))[: size - 1])
    lookup = {word: index for index, word in enumerate(words)}
    totals = {
        name: sum(
            max(0, len(sentence) - 1)
            for sentence in sentences
            if membership[tuple(sentence)] == name
        )
        for name in names
    }
    if min(totals.values()) == 0:
        raise ValueError("Need causal targets in every sentence-group partition")
    limits = dict(zip(names, (train_limit, validation_limit, test_limit), strict=True))
    selections: dict[str, IntArray] = {}
    parts: dict[str, Partition] = {}
    for name in names:
        count = totals[name]
        selected = np.arange(count, dtype=np.int64)
        if limits[name] and count > limits[name]:
            selected = np.sort(
                np.random.default_rng(1618).choice(count, limits[name], replace=False)
            )
        selections[name] = selected
        parts[name] = Partition(
            np.full((len(selected), window), -1, dtype=np.int64),
            np.zeros(len(selected), dtype=np.int64),
            np.zeros(len(selected), dtype=np.int64),
        )
    transitions = np.zeros((size, size), dtype=np.float64)
    target_counts = np.zeros(size, dtype=np.float64)
    cursors: dict[str, int] = dict.fromkeys(names, 0)
    written: dict[str, int] = dict.fromkeys(names, 0)
    for sentence in sentences:
        name = membership[tuple(sentence)]
        ids = np.array([lookup.get(word, 0) for word in sentence], dtype=np.int64)
        if name == "train":
            np.add.at(transitions, (ids[:-1], ids[1:]), 1)
            np.add.at(target_counts, ids[1:], 1)
        for position in range(1, len(ids)):
            row = written[name]
            if row < len(selections[name]) and cursors[name] == selections[name][row]:
                context = ids[max(0, position - window) : position]
                parts[name].contexts[row, -len(context) :] = context
                parts[name].targets[row] = ids[position]
                parts[name].groups[row] = gids[tuple(sentence)]
                written[name] += 1
            cursors[name] += 1
    unigram = (target_counts + 0.5) / (target_counts.sum() + 0.5 * size)
    dataset = Dataset(
        words, parts["train"], parts["validation"], parts["test"], transitions, unigram
    )
    dataset.save(output / "dataset.npz")
    write_json(
        output / "dataset.json",
        {
            "source": str(source),
            **(
                {"source_sha256": fingerprint(source)}
                if official is None
                else {
                    "source_files_sha256": {
                        filename: fingerprint(source / filename)
                        for filename in WIKITEXT_FILES.values()
                    }
                }
            ),
            "split_seed": seed if official is None else None,
            "split_policy": "normalized sentence groups"
            if official is None
            else "official WikiText",
            "partition_unit": "sentence" if official is None else "article",
            "sentences" if official is None else "articles": len(sentences),
            "sentence_groups" if official is None else "article_groups": len(groups),
            "vocabulary": size,
            "window": window,
            "eligible_targets": totals,
            "selected_targets": {name: len(parts[name].targets) for name in names},
            "unk_rate": {
                name: sum(int(target) == 0 for target in parts[name].targets)
                / len(parts[name].targets)
                for name in names
            },
            "statistics": "All training text; vocabulary/statistics never use held-out rows",
        },
    )
    return dataset
