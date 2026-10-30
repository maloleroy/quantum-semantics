"""Deterministic byte-pair tokens with complete UTF-8 byte coverage."""

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

CHUNKS = re.compile(r"\s+|[^\s]+")


def merge_pair(ids: tuple[int, ...], pair: tuple[int, int], token: int) -> tuple[int, ...]:
    merged: list[int] = []
    position = 0
    while position < len(ids):
        if position + 1 < len(ids) and (ids[position], ids[position + 1]) == pair:
            merged.append(token)
            position += 2
        else:
            merged.append(ids[position])
            position += 1
    return tuple(merged)


@dataclass(frozen=True)
class BytePairs:
    """ID zero remains UNK; IDs 1..256 cover bytes; later IDs are ranked merges."""

    size: int
    merges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.size < 512 or self.size & (self.size - 1) or len(self.merges) > self.size - 257:
            raise ValueError("Byte BPE needs a power-of-two register >=512 and fitting merges")
        for token, pair in enumerate(self.merges, start=257):
            if min(pair) < 1 or max(pair) >= token:
                raise ValueError("Merge operands must be earlier byte/merge IDs")
        if len(set(self.merges)) != len(self.merges):
            raise ValueError("Duplicate merge rules")

    @property
    def pieces(self) -> tuple[bytes, ...]:
        values = [b"", *(bytes([byte]) for byte in range(256))]
        for left, right in self.merges:
            values.append(values[left] + values[right])
        return tuple(values + [b""] * (self.size - len(values)))

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return tuple(
            "<unk>" if index == 0 else "hex:" + piece.hex() if piece else f"<unused:{index}>"
            for index, piece in enumerate(self.pieces)
        )

    def encode_chunk(self, text: str) -> tuple[int, ...]:
        ids = tuple(byte + 1 for byte in text.encode("utf-8"))
        ranks = {pair: token for token, pair in enumerate(self.merges, start=257)}
        while len(ids) > 1:
            token, pair = min(
                (ranks.get((a, b), self.size), (a, b)) for a, b in zip(ids, ids[1:], strict=False)
            )
            if token == self.size:
                break
            ids = merge_pair(ids, pair, token)
        return ids

    def encode(self, text: str) -> list[int]:
        return [token for chunk in CHUNKS.findall(text) for token in self.encode_chunk(chunk)]

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        pieces = self.pieces
        selected: list[bytes] = []
        for token in ids:
            if not 0 <= token < self.size or not pieces[token]:
                raise ValueError("UNK/unused outcomes have no byte decoding; do not discard them")
            selected.append(pieces[token])
        return b"".join(selected)


def fit_byte_pairs(texts: Iterable[str], size: int = 512) -> BytePairs:
    """Fit only on supplied training text; deterministic frequency/lexical tie break."""
    BytePairs(size, ())
    chunks = Counter(chunk for text in texts for chunk in CHUNKS.findall(text))
    if not chunks:
        raise ValueError("Need nonempty training text")
    sequences: Counter[tuple[int, ...]] = Counter(
        {
            tuple(byte + 1 for byte in chunk.encode("utf-8")): count
            for chunk, count in chunks.items()
        }
    )
    merges: list[tuple[int, int]] = []
    for token in range(257, size):
        pairs: Counter[tuple[int, int]] = Counter()
        for ids, count in sequences.items():
            for left, right in zip(ids, ids[1:], strict=False):
                pairs[left, right] += count
        if not pairs:
            break
        pair = min(pairs, key=lambda value: (-pairs[value], value))
        merges.append(pair)
        updated: Counter[tuple[int, ...]] = Counter()
        for ids, count in sequences.items():
            updated[merge_pair(ids, pair, token)] += count
        sequences = updated
    return BytePairs(size, tuple(merges))
