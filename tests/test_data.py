import numpy as np
import pytest

from qcse.data import DATASETS, load_corpus, load_phrases, make_examples, split_examples


@pytest.fixture
def corpora(tmp_path, monkeypatch):
    content = {
        "phrases": "First sentence.\nShared words.\nOnly original.\n",
        "cleaned": 'Cleaned_Sentence\nshared words\n"Quoted, words."\nCafé isn’t empty.\n'
        "Shared words!\nMeet at 42.\nGo https://example.com now.\nRare vocabulary.\n",
    }
    paths = {}
    for name, source in DATASETS.items():
        paths[name] = tmp_path / source.name
        paths[name].write_text(content[name], encoding="utf-8")
    # An old Tatoeba file can remain in a checkout without affecting words or IDs.
    (tmp_path / "tatoeba.csv").write_text("Excluded excluded tokens.\n", encoding="utf-8")
    monkeypatch.setattr("qcse.data.DATASETS", paths)
    return paths


def test_known_header_and_unicode(corpora):
    rows = load_phrases(corpora["cleaned"])
    assert rows[:3] == [["shared", "words"], ["quoted", "words"], ["café", "isn't", "empty"]]
    assert load_phrases(corpora["phrases"])[0] == ["first", "sentence"]


def test_fixed_global_vocabulary_across_ablations(corpora):
    full = load_corpus()
    single = load_corpus(datasets=["phrases"], cleaning="strict", max_sentences=2)
    assert full.vocabulary == single.vocabulary
    assert "rare" in single.vocabulary
    assert "cleaned" not in full.vocabulary
    assert "excluded" not in full.vocabulary
    assert full.summary["datasets"] == ["phrases", "cleaned"]
    assert full.summary["sentences_before_sampling"] == 8
    shared = full.sentences.index(["shared", "words"])
    assert full.sources[shared] == ["cleaned", "phrases"]
    assert len(load_corpus(cleaning="basic").sentences) == 10
    strict = load_corpus(cleaning="strict")
    assert len(strict.sentences) == 6
    assert "https" in strict.vocabulary  # Filtering examples never changes word IDs.
    custom = load_corpus(paths=[corpora["phrases"]])
    assert "rare" not in custom.vocabulary


@pytest.mark.parametrize("sampling", ["uniform", "balanced"])
@pytest.mark.parametrize("objective", ["causal", "cbow"])
def test_sampling_is_seeded_and_split_has_no_duplicate_leakage(corpora, sampling, objective):
    corpus = load_corpus(sampling=sampling, max_sentences=6, seed=7)
    assert corpus == load_corpus(sampling=sampling, max_sentences=6, seed=7)
    examples = make_examples(corpus.sentences, corpus.vocabulary, objective=objective)
    train, test = split_examples(examples, corpus.sentences)

    def groups(ids):
        return {tuple(corpus.sentences[examples[i].sentence]) for i in ids}

    assert not groups(train) & groups(test)
    assert len(np.r_[train, test]) == len(examples)


def test_balanced_sampling_redistributes_when_source_is_exhausted(corpora):
    corpus = load_corpus(sampling="balanced", max_sentences=7)
    assert len(corpus.sentences) == len(set(map(tuple, corpus.sentences))) == 7
    assert set().union(*map(set, corpus.sources)) == set(corpora)
    with pytest.raises(ValueError, match="at least 2"):
        load_corpus(max_sentences=1)
    with pytest.raises(ValueError, match="either"):
        load_corpus(paths=[corpora["phrases"]], datasets=["phrases"])
