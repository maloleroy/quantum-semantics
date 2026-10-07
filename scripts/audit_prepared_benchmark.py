"""Independent complete-array checks for a prepared benchmark, including historical studies."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from qcse.benchmark import fingerprint
from qcse.data import sentence_rows, tokenize
from qcse.outputs import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmark-audit/prepared-audit.json")
    )
    args = parser.parse_args()
    root = args.study
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["fingerprint"] == fingerprint(
        {k: v for k, v in manifest.items() if k != "fingerprint"}
    )
    for name, expected in manifest["artifact_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    for name, expected in manifest["code_sha256"].items():
        assert hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() == expected, name
    vocabulary = json.loads((root / "vocabulary.json").read_text())
    sentences = json.loads((root / "sentences.json").read_text())
    assert len({tuple(s) for s in sentences}) == len(sentences)
    counts = Counter()
    full_pool = set()
    for source in manifest["corpus"]["sources"]:
        path = Path(source["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
        for line in sentence_rows(path):
            words = tokenize(line)
            counts.update(words)
            if len(words) >= 2:
                full_pool.add(tuple(words))
    assert sorted(counts, key=lambda w: (-counts[w], w)) == vocabulary
    if manifest["settings"]["max_sentences"] is None:
        assert {tuple(s) for s in sentences} == full_pool
    lookup = {w: i for i, w in enumerate(vocabulary)}
    lengths = np.array([len(s) for s in sentences])
    offsets = np.r_[0, lengths.cumsum()[:-1]]
    tokens = np.array([lookup[w] for s in sentences for w in s])
    tracks, outer = [], []
    for objective in manifest["settings"]["objectives"]:
        with np.load(root / f"{objective}.npz") as data:
            contexts, targets, groups, positions = (
                data[k] for k in ("contexts", "targets", "groups", "positions")
            )
            n = len(targets)
            expected_count = int((lengths - 1).sum() if objective == "causal" else lengths.sum())
            assert n == expected_count
            window = manifest["settings"]["window"]
            relative = (
                np.arange(-window, 0)
                if objective == "causal"
                else np.r_[np.arange(-window // 2, 0), np.arange(1, window // 2 + 1)]
            )
            for start in range(0, n, 50000):
                batch = slice(start, start + 50000)
                g, p = groups[batch], positions[batch]
                assert np.array_equal(targets[batch], tokens[offsets[g] + p])
                expected_positions = p[:, None] + relative
                valid = (expected_positions >= 0) & (expected_positions < lengths[g, None])
                indices = offsets[g, None] + expected_positions
                expected = np.where(
                    valid, tokens[indices.clip(0, len(tokens) - 1)], len(vocabulary)
                )
                assert np.array_equal(contexts[batch], expected), (objective, start)
            dev, test = data["development"], data["test"]
            assert np.array_equal(np.sort(np.r_[dev, test]), np.arange(n))
            dev_groups, test_groups = set(groups[dev]), set(groups[test])
            assert not dev_groups & test_groups
            outer.append((dev_groups, test_groups))
            validated = []
            for fold in range(manifest["settings"]["folds"]):
                train, val = data[f"train_{fold}"], data[f"val_{fold}"]
                assert np.array_equal(np.sort(np.r_[train, val]), dev)
                assert not set(groups[train]) & set(groups[val])
                assert not set(groups[train]) & test_groups
                assert not set(groups[val]) & test_groups
                evaluated = data[f"val_eval_{fold}"]
                selected = set(groups[evaluated])
                assert np.array_equal(evaluated, val[np.isin(groups[val], list(selected))])
                validated.extend(set(groups[val]))
            assert len(set(validated)) == len(validated) == len(dev_groups)
            sample = data["test_eval"]
            selected = set(groups[sample])
            assert np.array_equal(sample, test[np.isin(groups[test], list(selected))])
            tracks.append(
                dict(
                    objective=objective,
                    examples=n,
                    folds=manifest["settings"]["folds"],
                    development_sentences=len(dev_groups),
                    test_sentences=len(test_groups),
                    evaluated_test_sentences=len(selected),
                    evaluated_test_examples=len(sample),
                )
            )
    assert all(value == outer[0] for value in outer)
    keys = [(t["objective"], t["arm"]["name"], t["seed"], t["fold"]) for t in manifest["tasks"]]
    assert len(set(keys)) == len(keys)
    result = dict(
        manifest=manifest["fingerprint"],
        sentences=len(sentences),
        vocabulary=len(vocabulary),
        tasks=len(keys),
        status="passed",
        tracks=tracks,
        scope=(
            "All target/context rows, all folds, full pool/vocabulary, artifact/source hashes; "
            "no training quality or CUDA claim"
        ),
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
