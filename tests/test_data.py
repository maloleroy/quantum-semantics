from pathlib import Path

import numpy as np
import pytest

from src.data import Dataset, load_sentences, load_wikitext, prepare, tokenize


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


def test_wikitext_preserves_official_articles_and_cross_line_context(tmp_path: Path) -> None:
    source = tmp_path / "wikitext"
    source.mkdir()
    (source / "wiki.train.raw").write_text(
        "\n = Train = \n\nalpha beta\ngamma delta\n = = Section = = \nepsilon\n"
        " = Other = \nzeta eta\n",
        encoding="utf-8",
    )
    (source / "wiki.valid.raw").write_text(" = Valid = \nheldoutword alpha\n", encoding="utf-8")
    (source / "wiki.test.raw").write_text(" = Test = \ntestword beta\n", encoding="utf-8")
    articles = load_wikitext(source)
    assert len(articles["train"]) == 2
    assert articles["train"][0] == [
        "train",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "section",
        "epsilon",
    ]
    destination = tmp_path / "prepared"
    destination.mkdir()
    dataset = prepare(
        source,
        destination,
        vocabulary=16,
        window=3,
        train_limit=0,
        validation_limit=0,
        test_limit=0,
    )
    assert "heldoutword" not in dataset.vocabulary
    assert "testword" not in dataset.vocabulary
    assert len(set(dataset.train.groups)) == 2
    lookup = {word: index for index, word in enumerate(dataset.vocabulary)}
    gamma = np.flatnonzero(dataset.train.targets == lookup["gamma"])[0]
    np.testing.assert_array_equal(
        dataset.train.contexts[gamma], [lookup.get(word, 0) for word in ("train", "alpha", "beta")]
    )
    # The new article begins with its own title and never carries the last one's context.
    np.testing.assert_array_equal(dataset.train.contexts[-2], [-1, -1, lookup.get("other", 0)])
    assert dataset.transitions.sum() == 8
    assert len(dataset.validation.targets) == 2
    assert len(dataset.test.targets) == 2
    Dataset.load(destination / "dataset.npz")
    # Normalization must not let an identical document enter another partition.
    (source / "wiki.test.raw").write_text(" = VALID = \nheldoutword alpha\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps"):
        prepare(source, destination)
