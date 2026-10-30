"""Lossless byte-backed WikiText preparation with training-only merge statistics."""

from pathlib import Path

import numpy as np

from .data import ARTICLE_TITLE, WIKITEXT_FILES, Dataset, IntArray, Partition, tokenize
from .storage import fingerprint, write_json
from .tokenization import CHUNKS, fit_byte_pairs


def raw_wikitext(source: Path) -> dict[str, list[str]]:
    partitions: dict[str, list[str]] = {}
    for name, filename in WIKITEXT_FILES.items():
        articles: list[str] = []
        current: list[str] = []
        # Preserve text/newlines within each article; omit only leading separator blanks.
        with (source / filename).open(encoding="utf-8", newline="") as file:
            for line in file:
                if ARTICLE_TITLE.fullmatch(line.strip()):
                    if current:
                        articles.append("".join(current))
                    current = [line]
                elif current:
                    current.append(line)
                elif line.strip():
                    raise ValueError(f"Expected an article title in {filename}")
        if current:
            articles.append("".join(current))
        if not articles:
            raise ValueError(f"No articles in {filename}")
        partitions[name] = articles
    return partitions


def prepare_subwords(
    source: Path,
    output: Path,
    vocabulary: int = 512,
    window: int = 128,
    train_limit: int = 32768,
    validation_limit: int = 4096,
    test_limit: int = 8192,
) -> Dataset:
    if window < 1 or min(train_limit, validation_limit, test_limit) < 0:
        raise ValueError("Need a positive window and nonnegative limits")
    texts = raw_wikitext(source)
    # Keep the old normalization-based leakage guard even though byte inputs are lossless.
    canonical = {text: tuple(tokenize(text)) for articles in texts.values() for text in articles}
    groups = sorted(set(canonical.values()))
    gids = {group: index for index, group in enumerate(groups)}
    membership: dict[tuple[str, ...], str] = {}
    for name, articles in texts.items():
        for text in articles:
            group = canonical[text]
            if group in membership and membership[group] != name:
                raise ValueError("Normalized article overlaps across official partitions")
            membership[group] = name
    codec = fit_byte_pairs(texts["train"], vocabulary)
    lengths = np.asarray([len(piece) for piece in codec.pieces], dtype=np.int64)
    cache: dict[str, tuple[int, ...]] = {}
    encoded: dict[str, list[tuple[int, IntArray]]] = {}
    for name, articles in texts.items():
        documents: list[tuple[int, IntArray]] = []
        for text in articles:
            token_ids: list[int] = []
            for chunk in CHUNKS.findall(text):
                if chunk not in cache:
                    cache[chunk] = codec.encode_chunk(chunk)
                token_ids.extend(cache[chunk])
            if codec.decode_bytes(token_ids) != text.encode("utf-8"):
                raise ValueError("Byte tokenizer failed exact article round trip")
            documents.append((gids[canonical[text]], np.asarray(token_ids, dtype=np.int64)))
        encoded[name] = documents
    totals = {name: sum(len(ids) - 1 for _, ids in docs) for name, docs in encoded.items()}
    if min(totals.values()) < 1:
        raise ValueError("Need causal targets in every official partition")
    transitions = np.zeros((vocabulary, vocabulary), dtype=np.float64)
    counts = np.zeros(vocabulary, dtype=np.float64)
    for _, ids in encoded["train"]:
        np.add.at(transitions, (ids[:-1], ids[1:]), 1)
        np.add.at(counts, ids[1:], 1)
    parts: dict[str, Partition] = {}
    offsets = np.arange(-window, 0, dtype=np.int64)
    for name, limit in zip(encoded, (train_limit, validation_limit, test_limit), strict=True):
        total = totals[name]
        selected = (
            np.sort(np.random.default_rng(1618).choice(total, limit, replace=False))
            if 0 < limit < total
            else np.arange(total, dtype=np.int64)
        )
        contexts = np.full((len(selected), window), -1, dtype=np.int64)
        targets = np.zeros(len(selected), dtype=np.int64)
        selected_groups = np.zeros(len(selected), dtype=np.int64)
        cursor = 0
        for group, ids in encoded[name]:
            end = cursor + len(ids) - 1
            start_row, end_row = (int(value) for value in np.searchsorted(selected, [cursor, end]))
            positions = selected[start_row:end_row] - cursor + 1
            indices = positions[:, None] + offsets
            contexts[start_row:end_row] = np.where(indices >= 0, ids[indices.clip(min=0)], -1)
            targets[start_row:end_row] = ids[positions]
            selected_groups[start_row:end_row] = group
            cursor = end
        parts[name] = Partition(contexts, targets, selected_groups, lengths[targets])
    unigram = (counts + 0.5) / (counts.sum() + 0.5 * vocabulary)
    dataset = Dataset(
        codec.vocabulary,
        parts["train"],
        parts["validation"],
        parts["test"],
        transitions,
        unigram,
        codec,
    )
    dataset.save(output / "dataset.npz")
    write_json(
        output / "dataset.json",
        {
            "source": str(source),
            "source_files_sha256": {
                name: fingerprint(source / name) for name in WIKITEXT_FILES.values()
            },
            "split_policy": "official WikiText; normalized article overlap rejected",
            "partition_unit": "article",
            "tokenizer": "byte-bpe-v1; separate whitespace/nonwhitespace chunks; no normalization",
            "tokenizer_fit": "all training articles only, independently of optimization limits",
            "vocabulary": vocabulary,
            "merges": len(codec.merges),
            "unused_outcomes": vocabulary - 257 - len(codec.merges),
            "window_tokens": window,
            "eligible_targets": totals,
            "selected_targets": {name: len(part.targets) for name, part in parts.items()},
            "unk_rate": {name: float((part.targets == 0).mean()) for name, part in parts.items()},
            "corpus_bytes": {
                name: sum(len(text.encode("utf-8")) for text in docs)
                for name, docs in texts.items()
            },
            "corpus_tokens": {
                name: sum(len(ids) for _, ids in docs) for name, docs in encoded.items()
            },
            "round_trip": "every article checked byte-for-byte",
            "boundaries": "markup/whitespace retained; first token not predicted; no EOS",
            "statistics": "all training targets only; no held-out counts",
            "scoring": "token NLL/sampling/argmax; target NLL divided by target byte lengths",
        },
    )
    return dataset
