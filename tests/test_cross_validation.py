import json

import numpy as np
import pytest
import torch

from qcse.cli import parser, run
from qcse.data import make_examples, split_examples, validation_folds
from qcse.model import QCSEModel
from qcse.state_cache import ContextStates
from qcse.training import TrainConfig, load_run, metrics, train

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")),
    pytest.param(
        "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="No MPS")
    ),
]


@pytest.mark.parametrize("objective", ["causal", "cbow"])
def test_grouped_outer_holdout_and_five_inner_folds(objective):
    vocabulary = ["a", "b", "c", "d", "e"]
    sentences = [[a, b] for a in vocabulary for b in vocabulary]
    sentences += sentences[:10]  # Cross-source duplicates must stay together.
    examples = make_examples(sentences, vocabulary, objective=objective)
    development, test = split_examples(examples, sentences)

    def groups(ids):
        return {tuple(sentences[examples[i].sentence]) for i in ids}

    assert len(groups(test)) == 5
    assert len(groups(development)) == 20
    validation_seen = []
    for train_ids, validation_ids in validation_folds(examples, sentences, development):
        assert len(groups(train_ids)) == 16  # 64% train, 16% val, 20% test.
        assert len(groups(validation_ids)) == 4
        assert not groups(train_ids) & groups(validation_ids)
        assert not (groups(train_ids) | groups(validation_ids)) & groups(test)
        validation_seen.extend(validation_ids)
    np.testing.assert_array_equal(sorted(validation_seen), sorted(development))
    first = list(validation_folds(examples, sentences, development))
    second = list(validation_folds(examples, sentences, development))
    for (a, b), (c, d) in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
    with pytest.raises(ValueError, match="distinct development"):
        list(validation_folds(examples, sentences, development[:2]))


@pytest.mark.parametrize("objective", ["causal", "cbow"])
@pytest.mark.parametrize("device", DEVICES)
def test_cli_cross_validation_and_resume(tmp_path, monkeypatch, objective, device):
    import qcse.cross_validation as cv

    sentences = [[a, b, "c"] for a in "abcd" for b in "abcd"]
    data = tmp_path / "sentences.csv"
    data.write_text("\n".join(" ".join(words) for words in sentences))
    output = tmp_path / "runs"
    actual_train = cv.train
    starts = []

    def interrupted(*args, **kwargs):
        starts.append(args[0].weights.copy())
        if len(starts) == 2:
            raise KeyboardInterrupt
        return actual_train(*args, **kwargs)

    monkeypatch.setattr(cv, "train", interrupted)
    command = [
        "train",
        "--data",
        str(data),
        "--output",
        str(output),
        "--folds",
        "5",
        "--epochs",
        "2",
        "--objective",
        objective,
        "--device",
        device,
    ]
    with pytest.raises(KeyboardInterrupt):
        run(parser().parse_args(command))
    folder = next(output.iterdir())
    first_checkpoint = (folder / "fold-01/run.npz").read_bytes()

    # The failed second fit and all subsequent fits start from the same weights.
    def track_starts(*args, **kwargs):
        starts.append(args[0].weights.copy())
        return actual_train(*args, **kwargs)

    monkeypatch.setattr(cv, "train", track_starts)
    calls = []

    def final_test_only(probabilities, targets):
        calls.append(len(targets))
        return metrics(probabilities, targets)

    monkeypatch.setattr(cv, "metrics", final_test_only)
    run(parser().parse_args(["resume-cv", str(folder), "--device", device]))
    assert len(starts) == 7
    for initial in starts[1:]:
        np.testing.assert_array_equal(initial, starts[0])
    assert (folder / "fold-01/run.npz").read_bytes() == first_checkpoint
    result = json.loads((folder / "cross_validation.json").read_text())
    assert result["status"] == "complete"
    assert len(result["folds"]) == 5
    assert calls == [result["test_examples"]]  # Held-out test is scored exactly once.
    run(parser().parse_args(["resume-cv", str(folder), "--device", device]))
    assert calls == [result["test_examples"]]  # Completed resumes do not score it again.
    aggregate = json.loads((folder / "cv_history.json").read_text())
    assert len(aggregate) == 3
    assert "mean" in aggregate[-1]["validation"]["bce"]
    for path in folder.glob("fold-*/run.npz"):
        saved = load_run(path)
        assert saved["state"]["epoch"] == 2
        assert len(saved["test_ids"]) == 0
        assert len(saved["validation_ids"]) > 0
        assert saved["metadata"]["evaluation"] == "validation"
        assert all("test" not in row for row in saved["state"]["history"])
    refit = load_run(folder / "refit/run.npz")
    assert refit["state"]["epoch"] == 2
    assert len(refit["train_ids"]) == result["development_examples"]
    assert all(set(row) == {"epoch", "train"} for row in refit["state"]["history"])
    with np.load(folder / "test_embeddings.npz") as archive:
        assert archive["probabilities"].shape[0] == result["test_examples"]
    # Standalone fold continuation retains validation semantics and vocabulary.
    run(
        parser().parse_args(
            [
                "continue",
                str(folder / "fold-01/run.npz"),
                "--epochs",
                "1",
                "--output",
                str(tmp_path / "continued"),
            ]
        )
    )
    continued = load_run(next((tmp_path / "continued").glob("*/run.npz")))
    assert continued["state"]["history"][-1]["validation"]
    assert "test" not in continued["state"]["history"][-1]
    assert continued["metadata"]["model"]["vocabulary"] == refit["metadata"]["model"]["vocabulary"]


def test_partial_cv_fit_resumes_to_target_with_identical_results(tmp_path, monkeypatch):
    import qcse.cross_validation as cv

    data = tmp_path / "sentences.csv"
    data.write_text("\n".join(f"{a} {b} c" for a in "abcd" for b in "abcd"))
    command = ["train", "--data", str(data), "--folds", "5", "--epochs", "2"]
    run(parser().parse_args(command + ["--output", str(tmp_path / "baseline")]))
    baseline = next((tmp_path / "baseline").iterdir())
    actual_train = cv.train

    def interrupt_after_checkpoint(*args, **kwargs):
        checkpoint = kwargs["checkpoint_callback"]

        def save_then_interrupt(state):
            checkpoint(state)
            if state["epoch"] == 1:
                raise KeyboardInterrupt

        kwargs["checkpoint_callback"] = save_then_interrupt
        return actual_train(*args, **kwargs)

    monkeypatch.setattr(cv, "train", interrupt_after_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        run(parser().parse_args(command + ["--output", str(tmp_path / "interrupted")]))
    resumed = next((tmp_path / "interrupted").iterdir())
    assert load_run(resumed / "fold-01/run.npz")["state"]["epoch"] == 1
    monkeypatch.setattr(cv, "train", actual_train)
    run(parser().parse_args(["resume-cv", str(resumed)]))
    for original in baseline.glob("*/run.npz"):
        expected = load_run(original)
        actual = load_run(resumed / original.relative_to(baseline))
        assert actual["state"]["epoch"] == 2
        assert actual["state"]["history"] == expected["state"]["history"]
        assert actual["state"]["rng_state"] == expected["state"]["rng_state"]
        np.testing.assert_array_equal(actual["weights"], expected["weights"])


@pytest.mark.parametrize("device", DEVICES)
def test_streamed_contexts_match_resident_cache_and_remain_bounded(device):
    vocabulary = [f"word-{i}" for i in range(2**14)]
    sentences = [vocabulary[i : i + 3] for i in range(6)]
    examples = make_examples(sentences, vocabulary)
    streamed = QCSEModel(vocabulary, simulation_batch_size=3, device=device)
    resident = QCSEModel(vocabulary, simulation_batch_size=3, device=device)
    cache = ContextStates(streamed, examples, cache_mib=1)
    assert cache.packed is None
    weights = np.stack((streamed.weights, streamed.weights + 0.1))
    actual = cache.predict(np.arange(len(examples)), weights)
    expected = resident.predict_encoded([resident.encode(e.context) for e in examples], weights)
    tolerance = 2e-5 if device != "cpu" else 1e-12
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    assert len(cache.cache) <= cache.capacity == 4
    train_ids, test_ids = split_examples(examples, sentences)
    config = TrainConfig(epochs=2, batch_size=3)
    _, actual = train(streamed, examples, train_ids, test_ids, config, state_cache_mib=1)
    _, expected = train(resident, examples, train_ids, test_ids, config)
    np.testing.assert_allclose(streamed.weights, resident.weights, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
