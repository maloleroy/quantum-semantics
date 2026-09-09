"""Sentence loading, deterministic vocabulary and sentence-local examples."""

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "phrases.csv"
TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower().replace("’", "'"))


def load_phrases(path: Path = DEFAULT_DATA) -> list[list[str]]:
    """Read headerless lines (including unquoted commas), or a one-column CSV.

    No header is inferred: the supplied corpus has none. Do not treat the first
    phrase as column names. A fully quoted CSV field is unquoted with csv.reader.
    """
    sentences = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith('"') and line.endswith('"'):
            fields = next(csv.reader([line]))
            if len(fields) == 1:
                line = fields[0]
        words = tokenize(line)
        if words:
            sentences.append(words)
    if not sentences:
        raise ValueError(f"No words found in {path}")
    return sentences


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
