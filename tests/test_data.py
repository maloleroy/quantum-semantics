from pathlib import Path

import numpy as np

from src.data import Dataset, load_sentences, prepare, tokenize


def test_tokenization_headers_and_train_only_grouped_data(tmp_path: Path) -> None:
    source = tmp_path / "sentences.csv"
    lines = [
        f"{color} {animal} sees the {place}"
        for color in ("red", "blue", "green", "white")
        for animal in ("bird", "cat", "fish")
        for place in ("tree", "sky")
    ]
    source.write_text("text\n" + "\n".join(lines + [lines[0]]) + "\n")
    assert tokenize("Aren’t birds—lovely?") == ["aren't", "birds", "lovely"]
    assert len(load_sentences(source)) == 25
    output = tmp_path / "prepared"
    output.mkdir()
    dataset = prepare(source, output, vocabulary=8, train_limit=3, validation_limit=0, test_limit=0)
    assert len(dataset.train.targets) == 3
    assert dataset.transitions.sum() > len(dataset.train.targets)
    groups = sorted(set(map(tuple, load_sentences(source))))
    rng = np.random.default_rng(20261026)
    order = rng.permutation(len(groups))
    held_out = set(order[: round(0.2 * len(groups)) + round(0.16 * len(groups))])
    training_words = {
        word for index, sentence in enumerate(groups) if index not in held_out for word in sentence
    }
    assert set(dataset.vocabulary[1:]) <= training_words
    restored = Dataset.load(output / "dataset.npz")
    assert restored.vocabulary == dataset.vocabulary
    np.testing.assert_array_equal(restored.train.contexts, dataset.train.contexts)
    assert not set(restored.train.groups) & set(restored.test.groups)


def test_vocabulary_statistics_are_independent_of_optimization_limit(tmp_path: Path) -> None:
    source = tmp_path / "sentences.csv"
    source.write_text(
        "\n".join(
            f"{a} sees {b} near home"
            for a in ("alice", "bob", "carol", "david")
            for b in ("trees", "birds", "cats", "dogs")
        )
    )
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    small = prepare(source, first, vocabulary=8, train_limit=2)
    full = prepare(source, second, vocabulary=8, train_limit=0)
    assert small.vocabulary == full.vocabulary
    np.testing.assert_array_equal(small.transitions, full.transitions)
    np.testing.assert_allclose(small.unigram, full.unigram)
