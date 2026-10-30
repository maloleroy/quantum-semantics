from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.data import Dataset
from src.generation import encode_prompt, generate
from src.metrics import evaluate
from src.models import ModelConfig, build_model
from src.storage import save_arrays
from src.subwords import prepare_subwords, raw_wikitext
from src.tokenization import CHUNKS, BytePairs, fit_byte_pairs, merge_pair
from src.training import TrainConfig, selected_models, train


def corpus(path: Path) -> Path:
    source = path / "corpus"
    source.mkdir()
    (source / "wiki.train.raw").write_text(
        "\n = Train = \nA cat, a CAT: 123.\n\n = Second = \nBananas!\n", encoding="utf-8"
    )
    (source / "wiki.valid.raw").write_text(" = Valid = \nUnseen: 日本語 🦊.\n", encoding="utf-8")
    (source / "wiki.test.raw").write_text(" = Test = \nAnother word—42!\n", encoding="utf-8")
    return source


def test_byte_pairs_round_trip_unseen_utf8_and_ranked_merge_reference() -> None:
    codec = fit_byte_pairs(["aaaa abab banana banana CAT 12!\n"], 512)
    assert codec == fit_byte_pairs(["aaaa abab banana banana CAT 12!\n"], 512)
    for text in ("aaaaa ababab", "Unseen 日本語 🦊\n\t 123—CAT!", "\x00é", ""):
        ids = codec.encode(text)
        assert 0 not in ids
        assert codec.decode_bytes(ids) == text.encode("utf-8")
        reference: list[int] = []
        for chunk in CHUNKS.findall(text):
            initial = tuple(byte + 1 for byte in chunk.encode("utf-8"))
            for token, pair in enumerate(codec.merges, start=257):
                initial = merge_pair(initial, pair, token)
            reference.extend(initial)
        assert ids == reference
    with pytest.raises(ValueError, match="UNK/unused"):
        codec.decode_bytes([0])
    with pytest.raises(ValueError, match="earlier"):
        BytePairs(512, ((1, 257),))


def test_byte_data_preserves_articles_and_training_only_statistics(tmp_path: Path) -> None:
    source = corpus(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    full = prepare_subwords(
        source, first, window=8, train_limit=0, validation_limit=0, test_limit=0
    )
    codec = full.byte_pairs
    assert codec is not None
    texts = raw_wikitext(source)
    assert codec.decode_bytes(codec.encode(texts["train"][0])) == texts["train"][0].encode()
    assert "CAT: 123." in texts["train"][0]
    assert codec == fit_byte_pairs(texts["train"], 512)
    first_ids, next_ids = (codec.encode(text) for text in texts["train"])
    boundary = len(first_ids) - 1
    assert full.train.targets[boundary] == next_ids[1]
    np.testing.assert_array_equal(full.train.contexts[boundary], [-1] * 7 + [next_ids[0]])
    assert full.transitions.sum() == len(first_ids) + len(next_ids) - 2
    restored = Dataset.load(first / "dataset.npz")
    assert restored.byte_pairs == codec
    np.testing.assert_array_equal(restored.train.target_bytes, full.train.target_bytes)
    small = prepare_subwords(
        source, second, window=8, train_limit=3, validation_limit=0, test_limit=0
    )
    assert small.byte_pairs == full.byte_pairs
    np.testing.assert_array_equal(small.transitions, full.transitions)
    np.testing.assert_array_equal(small.unigram, full.unigram)
    np.testing.assert_array_equal(small.validation.contexts, full.validation.contexts)
    (source / "wiki.valid.raw").write_text(
        " = New Valid = \nNever-seen 🧬🧬🧬.\n", encoding="utf-8"
    )
    changed = prepare_subwords(source, second, window=8, train_limit=3)
    assert changed.byte_pairs == full.byte_pairs
    np.testing.assert_array_equal(changed.transitions, full.transitions)
    np.testing.assert_array_equal(changed.unigram, full.unigram)
    (source / "wiki.test.raw").write_text(" = NEW VALID = \nNever-seen 🧬🧬🧬.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps"):
        prepare_subwords(source, second)


def test_byte_metrics_and_nontext_generation_do_not_renormalize(tmp_path: Path) -> None:
    source = corpus(tmp_path)
    dataset = prepare_subwords(source, tmp_path, window=8, train_limit=3, validation_limit=8)
    model = build_model(ModelConfig("unigram", 512, 8), dataset, 11)
    measured, arrays = evaluate(model, dataset.validation)
    losses = np.asarray(arrays["nll"], dtype=np.float64)
    lengths = np.asarray(arrays["target_bytes"], dtype=np.int64)
    expected = losses.sum() / lengths.sum() / np.log(2)
    assert measured.target_bits_per_byte == pytest.approx(float(expected))
    assert measured.unk_rate == 0
    assert measured.known_success == measured.success
    assert dataset.byte_pairs is not None
    assert encode_prompt(dataset, "Never seen: 雨42") == dataset.byte_pairs.encode(
        "Never seen: 雨42"
    )
    probabilities = np.zeros(512, dtype=np.float64)
    probabilities[0] = 1
    forced = replace(dataset, unigram=probabilities)
    invalid = build_model(ModelConfig("unigram", 512, 8), forced, 11)
    result = generate(forced, invalid, "CAT: 123!", steps=3)
    assert result.text == "CAT: 123!<unk><unk><unk>"
    assert result.tokens == (0, 0, 0) and result.non_text_outcomes == 3
    transitions = np.zeros((512, 512), dtype=np.float64)
    transitions[ord("q") + 1, 196] = 1000
    transitions[196, 170] = 1000
    chained = replace(dataset, transitions=transitions)
    chain = build_model(ModelConfig("bigram", 512, 8), chained, 11)
    decoded = generate(chained, chain, "q", steps=2, argmax=True)
    assert decoded.text == "qé" and decoded.tokens == (196, 170)
    # This count-only checkpoint has no optimizer; exercise tokenizer/metric persistence.
    run = train(
        tmp_path / "dataset.npz",
        [ModelConfig("unigram", 512, 8)],
        TrainConfig(epochs=1),
        [11],
        tmp_path,
    )
    restored, records = selected_models(run)
    assert restored.byte_pairs == dataset.byte_pairs
    assert records[0]["validation"]["target_bits_per_byte"] == measured.target_bits_per_byte
    with np.load(tmp_path / "dataset.npz") as saved:
        altered = {key: saved[key] for key in saved.files}
    altered["validation_target_bytes"] = np.ones(8, dtype=np.int64) * 999
    save_arrays(tmp_path / "altered.npz", altered)
    with pytest.raises(ValueError, match="disagree"):
        Dataset.load(tmp_path / "altered.npz")
