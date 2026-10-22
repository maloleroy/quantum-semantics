"""Sentence loading, deterministic vocabulary and sentence-local examples."""

import csv
import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "phrases.csv"
DATASETS = {
    "phrases": DEFAULT_DATA,
    "cleaned": DEFAULT_DATA.with_name("cleaned_sentences.csv"),
}
HEADERS = {"sentence", "sentences", "cleaned_sentence", "text"}
TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(unicodedata.normalize("NFKC", text).lower().replace("’", "'"))


def sentence_rows(path: Path):
    """Read sentence lines or one-column CSV, skipping an explicit known header.

    Unquoted commas belong to the sentence. Only the first nonempty row can be
    a header; ordinary first sentences are retained.
    """
    first = True
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            if line.startswith('"') and line.endswith('"'):
                fields = next(csv.reader([line]))
                if len(fields) == 1:
                    line = fields[0]
            if first and line.strip().lower() in HEADERS:
                first = False
                continue
            first = False
            yield line


def load_phrases(path: Path = DEFAULT_DATA) -> list[list[str]]:
    sentences = [words for line in sentence_rows(path) if (words := tokenize(line))]
    if not sentences:
        raise ValueError(f"No words found in {path}")
    return sentences


@dataclass
class Corpus:
    sentences: list[list[str]]
    vocabulary: list[str]
    sources: list[list[str]]
    summary: dict


def load_corpus(
    *, paths=None, datasets=None, cleaning="dedupe", sampling="uniform", max_sentences=None, seed=42
) -> Corpus:
    """Keep a vocabulary from every source, then filter/sample training sentences.

    Named dataset ablations share the vocabulary of phrases and cleaned sentences.
    Explicit paths instead define a custom corpus and its full vocabulary.
    """
    if cleaning not in ("basic", "dedupe", "strict"):
        raise ValueError("cleaning must be basic, dedupe or strict")
    if sampling not in ("uniform", "balanced"):
        raise ValueError("sampling must be uniform or balanced")
    if max_sentences is not None and max_sentences < 2:
        raise ValueError("max-sentences must be at least 2")
    if paths is not None and datasets is not None:
        raise ValueError("Choose either custom data paths or named datasets")
    files = {str(Path(p).resolve()): Path(p) for p in paths} if paths else DATASETS
    selected = set(datasets if datasets is not None else files)
    if not selected or not selected <= files.keys():
        raise ValueError("Select at least one known dataset")
    counts = Counter()
    sentences, sources, source_stats = [], [], []
    seen = {}
    for name, path in files.items():
        rows = usable = filtered = duplicates = 0
        for text in sentence_rows(path):
            rows += 1
            words = tokenize(text)
            counts.update(words)
            if name not in selected:
                continue
            if len(words) < 2 or (
                cleaning == "strict"
                and (len(words) > 40 or re.search(r"\d|https?://|www\.|@", text, re.I))
            ):
                filtered += 1
                continue
            usable += 1
            key = tuple(words)
            if cleaning != "basic" and key in seen:
                duplicates += 1
                sources[seen[key]].add(name)
                continue
            seen[key] = len(sentences)
            sentences.append(words)
            sources.append({name})
        source_stats.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": rows,
                "selected": name in selected,
                "usable_rows": usable,
                "filtered_rows": filtered,
                "duplicates_merged": duplicates,
            }
        )
    available = len(sentences)
    if max_sentences is not None and max_sentences < available:
        rng = np.random.default_rng(seed)
        if sampling == "uniform":
            indices = sorted(rng.choice(available, max_sentences, replace=False).tolist())
        else:
            # Round-robin sources; a shared sentence occupies only one sample slot.
            pools = [
                iter(
                    rng.permutation(
                        [i for i, origin in enumerate(sources) if name in origin]
                    ).tolist()
                )
                for name in files
                if name in selected
            ]
            chosen = set()
            while len(chosen) < max_sentences:
                for pool in pools:
                    index = next((i for i in pool if i not in chosen), None)
                    if index is not None:
                        chosen.add(index)
                    if len(chosen) == max_sentences:
                        break
            indices = sorted(chosen)
        sentences = [sentences[i] for i in indices]
        sources = [sources[i] for i in indices]
    if len(sentences) < 2:
        raise ValueError("At least two usable sentences are required after cleaning")
    vocabulary = sorted(counts, key=lambda word: (-counts[word], word))
    return Corpus(
        sentences,
        vocabulary,
        [sorted(origin) for origin in sources],
        {
            "sources": source_stats,
            "datasets": [name for name in files if name in selected],
            "cleaning": cleaning,
            "sampling": sampling,
            "max_sentences": max_sentences,
            "sentences_before_sampling": available,
            "vocabulary_scope": "all source rows before cleaning, sampling and splitting",
            "vocabulary_tokens": sum(counts.values()),
        },
    )


def build_vocabulary(sentences: list[list[str]]) -> list[str]:
    """Frequent words first; alphabetical tie break; IDs are zero-based."""
    counts = Counter(word for sentence in sentences for word in sentence)
    return sorted(counts, key=lambda word: (-counts[word], word))


def word_bits(ids, qubits: int) -> np.ndarray:
    """Columns are q0, q1, ... (least-significant bit first, as in Qiskit)."""
    ids = np.asarray(ids, dtype=np.int64)
    return ((ids[..., None] >> np.arange(qubits)) & 1).astype(float)


@dataclass(frozen=True)
class Example:
    sentence: int
    position: int
    target: int
    context: tuple[int, ...]


def make_examples(sentences, vocabulary, window: int = 4, objective: str = "cbow") -> list[Example]:
    """Build sentence-local CBOW or causal next-token examples."""
    if objective not in ("cbow", "causal"):
        raise ValueError("objective must be 'cbow' or 'causal'")
    if window < 1 or (objective == "cbow" and window % 2):
        raise ValueError("window must be positive; CBOW requires an even total context size")
    lookup = {word: i for i, word in enumerate(vocabulary)}
    result = []
    for sid, words in enumerate(sentences):
        try:
            ids = [lookup[word] for word in words]
        except KeyError as error:
            raise ValueError(f"Word outside the saved vocabulary: {error.args[0]!r}") from error
        for pos, target in enumerate(ids):
            if objective == "causal":
                context = tuple(ids[max(0, pos - window) : pos])
            else:
                radius = window // 2
                context = tuple(ids[max(0, pos - radius) : pos] + ids[pos + 1 : pos + radius + 1])
            if context:
                result.append(Example(sid, pos, target, context))
    return result


def split_examples(examples, sentences, seed: int = 42, test_fraction: float = 0.2):
    """Split by sentence text, keeping duplicate phrases in the same partition."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie between zero and one")
    groups = sorted({tuple(sentences[e.sentence]) for e in examples})
    if len(groups) < 2:
        raise ValueError("At least two distinct usable sentences are required for training")
    order = np.random.default_rng(seed).permutation(len(groups))
    count = min(len(groups) - 1, max(1, round(len(groups) * test_fraction)))
    held_out = {groups[i] for i in order[:count]}
    test = np.array([tuple(sentences[e.sentence]) in held_out for e in examples])
    return np.flatnonzero(~test), np.flatnonzero(test)


def validation_folds(examples, sentences, development_ids, folds=5, seed=42, val_fraction=None):
    """Partition development sentence groups into train/val splits.

    val_fraction=None (default): k-fold rotation; each group validates exactly once.
    val_fraction set: repeated shuffle-split at that fraction; folds are independent runs.
    """
    if folds < 2:
        raise ValueError("Cross-validation requires at least two folds")
    development_ids = np.asarray(development_ids, dtype=np.int64)
    keys = [tuple(sentences[examples[i].sentence]) for i in development_ids]
    groups = sorted(set(keys))
    rng = np.random.default_rng(seed)
    if val_fraction is None:
        if len(groups) < folds:
            raise ValueError(
                f"Need at least {folds} distinct development sentences for {folds} folds"
            )
        order = rng.permutation(len(groups))
        for group_ids in np.array_split(order, folds):
            held_out = {groups[i] for i in group_ids}
            mask = np.array([key in held_out for key in keys])
            yield development_ids[~mask], development_ids[mask]
    else:
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must lie between zero and one")
        count = min(len(groups) - 1, max(1, round(len(groups) * val_fraction)))
        for _ in range(folds):
            order = rng.permutation(len(groups))
            held_out = {groups[i] for i in order[:count]}
            mask = np.array([key in held_out for key in keys])
            yield development_ids[~mask], development_ids[mask]
