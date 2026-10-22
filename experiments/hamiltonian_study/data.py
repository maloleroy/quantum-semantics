"""Sentence-group splitting, training vocabulary, and reproducible causal examples."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from qcse.data import load_phrases


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def prepare(
    path,
    output,
    vocabulary_size=512,
    window=8,
    train_limit=8192,
    val_limit=4096,
    test_limit=8192,
    split_seed=42,
):
    """Statistics use every training sentence, optimization may use a fixed subset."""
    sentences = load_phrases(Path(path))
    groups = sorted(set(map(tuple, sentences)))
    order = np.random.default_rng(split_seed).permutation(len(groups))
    n_test, n_val = round(0.2 * len(groups)), round(0.16 * len(groups))
    membership = {
        groups[j]: ("test" if i < n_test else "validation" if i < n_test + n_val else "train")
        for i, j in enumerate(order)
    }
    group_ids = {s: i for i, s in enumerate(groups)}
    counts = Counter(w for s in sentences if membership[tuple(s)] == "train" for w in s)
    words = sorted(counts, key=lambda w: (-counts[w], w))
    vocabulary = ["<unk>"] + words[: vocabulary_size - 1] if vocabulary_size else ["<unk>"] + words
    lookup = {w: i for i, w in enumerate(vocabulary)}
    size = len(vocabulary)
    transitions = np.zeros((size, size))
    cooc = np.zeros((size, size))
    target_counts = np.zeros(size)
    rows = {s: [] for s in ["train", "validation", "test"]}
    coverage = {s: [0, 0] for s in rows}
    for sentence in sentences:
        split = membership[tuple(sentence)]
        ids = [lookup.get(w, 0) for w in sentence]
        coverage[split][0] += sum(i != 0 for i in ids)
        coverage[split][1] += len(ids)
        for pos in range(1, len(ids)):
            context = ids[max(0, pos - window) : pos]
            rows[split].append((context, ids[pos], group_ids[tuple(sentence)], pos))
            if split == "train":
                target_counts[ids[pos]] += 1
                transitions[ids[pos - 1], ids[pos]] += 1
                for age, token in enumerate(reversed(context)):
                    cooc[token, ids[pos]] += 1 / (age + 1)
    payload = {
        "transitions": transitions,
        "cooc": cooc,
        "unigram": (target_counts + 0.5) / (target_counts.sum() + 0.5 * size),
        "vocabulary": np.array(vocabulary),
    }
    manifest = {
        "source": str(Path(path).resolve()),
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "sentences": len(sentences),
        "unique_sentence_groups": len(groups),
        "vocabulary_size": size,
        "training_word_types": len(counts),
        "window": window,
        "split_seed": split_seed,
        "partitions": {},
    }
    for split, limit in [("train", train_limit), ("validation", val_limit), ("test", test_limit)]:
        total = len(rows[split])
        selected = np.arange(total)
        if limit and total > limit:
            selected = np.sort(np.random.default_rng(1618).choice(total, limit, replace=False))
        context = np.full((len(selected), window), -1, dtype=np.int64)
        targets, gids, positions = [], [], []
        for i, j in enumerate(selected):
            c, y, g, pos = rows[split][j]
            context[i, -len(c) :] = c
            targets.append(y)
            gids.append(g)
            positions.append(pos)
        payload.update(
            {
                f"{split}_contexts": context,
                f"{split}_targets": np.array(targets),
                f"{split}_groups": np.array(gids),
                f"{split}_positions": np.array(positions),
                f"{split}_indices": selected,
            }
        )
        manifest["partitions"][split] = {
            "full_targets": total,
            "evaluated_or_optimized_targets": len(selected),
            "sentence_groups": len({group_ids[s] for s in groups if membership[s] == split}),
            "known_token_coverage": coverage[split][0] / coverage[split][1],
            "sample_unk_target_rate": float(np.mean(np.array(targets) == 0)),
        }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "dataset.npz", **payload)
    write_json(output / "dataset.json", manifest)
    return manifest


def graph_matrix(cooc, kind="count", neighbors=24):
    """Symmetrized threshold-kNN; ties are kept, so pruning commutes with relabeling."""
    w = cooc + cooc.T
    np.fill_diagonal(w, 0)
    if kind in ("ppmi", "npmi"):
        total = w.sum()
        denom = w.sum(1)[:, None] * w.sum(0)[None, :]
        pmi = np.log(np.maximum(w * total, 1e-30) / np.maximum(denom, 1e-30))
        if kind == "npmi":
            pmi /= np.maximum(-np.log(np.maximum(w / total, 1e-30)), 1e-10)
        w = np.where(w > 0, np.maximum(pmi, 0), 0)
    if neighbors and neighbors < len(w):
        thresholds = np.partition(w, -neighbors, axis=1)[:, -neighbors]
        keep = (w >= thresholds[:, None]) & (w > 0)
        w = np.where(keep | keep.T, w, 0)
    degree = w.sum(1)
    inv = np.zeros_like(degree)
    np.divide(1, np.sqrt(degree), out=inv, where=degree > 0)
    return w * inv[:, None] * inv[None, :], w


def transition_probabilities(counts, unigram, strength=10.0):
    return (counts + strength * unigram) / (counts.sum(1, keepdims=True) + strength)
