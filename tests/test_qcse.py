import json

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qcse.circuit import ansatz_circuit, encoding_circuit
from qcse.cli import main
from qcse.context import METHODS, ContextConfig, context_matrix, encoding_angles
from qcse.data import (
    build_vocabulary,
    load_phrases,
    make_examples,
    split_examples,
    tokenize,
    word_bits,
)
from qcse.model import QCSEModel
from qcse.training import TrainConfig, metrics, train


def test_headerless_csv_and_boundaries(tmp_path):
    path = tmp_path / "phrases.csv"
    path.write_text('The river, slowly moves.\n"Cold wind blows."\nAlone\n')
    sentences = load_phrases(path)
    assert sentences[0] == ["the", "river", "slowly", "moves"]
    vocab = build_vocabulary(sentences)
    examples = make_examples(sentences, vocab)
    assert len(examples) == 7
    middle = examples[1]
    assert [vocab[i] for i in middle.context] == ["the", "slowly", "moves"]
    assert examples[0].context == tuple(vocab.index(w) for w in ["river", "slowly"])
    assert tokenize("Don't—stop! 42") == ["don't", "stop"]


def test_causal_examples_only_use_previous_words():
    sentences = [["one", "two", "three", "four"]]
    vocabulary = build_vocabulary(sentences)
    examples = make_examples(sentences, vocabulary, window=2, objective="causal")
    assert [[vocabulary[i] for i in example.context] for example in examples] == [
        ["one"],
        ["one", "two"],
        ["two", "three"],
    ]
    assert [vocabulary[example.target] for example in examples] == ["two", "three", "four"]


def test_split_keeps_identical_sentences_together():
    sentences = [["a", "b"], ["c", "a"], ["a", "b"], ["d", "e"]]
    examples = make_examples(sentences, build_vocabulary(sentences))
    train_ids, test_ids = split_examples(examples, sentences)
    train_groups = {tuple(sentences[examples[i].sentence]) for i in train_ids}
    test_groups = {tuple(sentences[examples[i].sentence]) for i in test_ids}
    assert not train_groups & test_groups
    assert len(train_ids) + len(test_ids) == len(examples)


@pytest.mark.parametrize("method", METHODS)
def test_context_equations(method):
    ids, size = [1, 3, 2], 7
    cfg = ContextConfig(method, alpha=0.4, omega=0.8, delta=0.7, prime=5, hash_size=11)
    actual = context_matrix(ids, size, cfg)
    theta = np.array(ids) * 2 * np.pi / size
    if method == "angular":
        np.testing.assert_allclose(actual, [[0.8 * theta[0], 0.8 * theta[1]], [0.8 * theta[2], 0]])
        return
    for i in range(3):
        for j in range(3):
            decay = np.exp(-0.4 * abs(i - j))
            if method == "exponential":
                expected = decay * np.sin(0.8 * theta[i]) * np.cos(0.8 * theta[j]) + theta[i]
            elif method == "diagonal":
                expected = np.log1p(ids[i]) if i == j else decay * np.sin(0.8 * theta[i]) + theta[i]
            elif method == "phase":
                expected = decay * np.sin(0.8 * i + 0.7 * theta[i]) + theta[i]
            else:
                expected = decay * np.sin(0.8 * i + (ids[i] * 5) % 11) + theta[i]
            assert actual[i, j] == pytest.approx(expected)


def test_padding_preserves_features_and_order():
    matrix = np.arange(25).reshape(5, 5)
    angles = encoding_angles(matrix, 3)
    assert angles.shape == (5, 3, 2)
    np.testing.assert_equal(angles.ravel()[:25], matrix.ravel())
    np.testing.assert_equal(angles.ravel()[25:], 0)


def test_encoding_state_against_independent_matrices():
    angles = np.array([[[0.4, 0.8], [0.2, -0.7]], [[-0.1, 0.9], [0.6, 0.5]]])

    def rx(t):
        return np.array(
            [[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]]
        )

    def rz(t):
        return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])

    # Qiskit basis order |q1 q0>; CX control q0 flips q1: 1 <-> 3.
    cx = np.eye(4)[[0, 3, 2, 1]]
    expected = np.ones(4, dtype=complex) / 2
    for layer in angles:
        expected = (
            cx
            @ np.kron(rz(layer[1, 1]) @ rx(layer[1, 0]), rz(layer[0, 1]) @ rx(layer[0, 0]))
            @ expected
        )
    actual = Statevector.from_instruction(encoding_circuit(angles)).data
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    circuit = encoding_circuit(angles)
    assert circuit.count_ops() == {"h": 2, "rx": 4, "rz": 4, "cx": 2}


def test_ansatz_state_against_independent_crz_matrix():
    circuit, parameters = ansatz_circuit(2, 1)
    # Only CRZ nonzero: in little endian the control-q0=1 states are indices 1, 3.
    values = [0, 0, 0, 0, 0.73]
    bound = circuit.assign_parameters(dict(zip(parameters, values, strict=True)))
    state = Statevector(np.ones(4) / 2).evolve(bound)
    np.testing.assert_allclose(state.data, np.array([1, np.exp(-0.365j), 1, np.exp(0.365j)]) / 2)
    reverse, ps = ansatz_circuit(2, 1, "reverse")
    state = Statevector(np.ones(4) / 2).evolve(
        reverse.assign_parameters(dict(zip(ps, values, strict=True)))
    )
    np.testing.assert_allclose(state.data, np.array([1, 1, np.exp(-0.365j), np.exp(0.365j)]) / 2)
    many, ps = ansatz_circuit(7, 3)
    assert len(ps) == 60
    assert many.count_ops()["crz"] == 18


def test_cli_layers_configures_ansatz(tmp_path, monkeypatch):
    data = tmp_path / "phrases.csv"
    data.write_text("a b c\nb c a\n")
    out = tmp_path / "run"
    monkeypatch.setattr(
        "sys.argv",
        ["qcse", "prepare", "--data", str(data), "--output", str(out), "--layers", "3"],
    )
    main()
    out = next(out.iterdir())
    summary = json.loads((out / "summary.json").read_text())
    assert summary["ansatz_layers"] == 3
    assert summary["trainable_parameters"] == 3 * (summary["qubits"] * 3 - 1)


def test_bit_order_and_marginals():
    model = QCSEModel(["a", "b", "c", "d"])
    model.weights[:] = 0
    ids = np.arange(4)
    expected = [[0, 0], [1, 0], [0, 1], [1, 1]]
    np.testing.assert_equal(word_bits(ids, 2), expected)
    states = [Statevector.from_int(int(i), 4) for i in ids]
    np.testing.assert_allclose(model.predict_encoded(states), expected)
    p = model.predict_encoded([model.encode([0, 2, 3])])[0]
    full = Statevector.from_instruction(model.circuit([0, 2, 3]))
    np.testing.assert_allclose(p, [full.probabilities([q])[1] for q in range(2)])
    assert np.linalg.norm(full.data) == pytest.approx(1)


def test_save_load_and_unknown_words(tmp_path):
    model = QCSEModel(["a", "b", "c"], context=ContextConfig("phase"), direction="reverse")
    path = tmp_path / "model.npz"
    model.save(path)
    restored = QCSEModel.load(path)
    assert model.embed_phrase("a b c") == restored.embed_phrase("a b c")
    with pytest.raises(ValueError, match="outside"):
        restored.embed_phrase("a unknown")


def test_causal_model_predicts_and_persists_objective(tmp_path):
    model = QCSEModel(["a", "b", "c"], objective="causal", window=2)
    prediction = model.predict_next("a b")
    assert prediction["context"] == ["a", "b"]
    path = tmp_path / "causal.npz"
    model.save(path)
    restored = QCSEModel.load(path)
    assert restored.objective == "causal"
    assert restored.predict_next("a b")["context"] == ["a", "b"]


def test_training_changes_weights_reproducibly():
    sentences = [["a", "b", "a"], ["b", "a", "b"], ["a", "a", "b"]]
    examples = make_examples(sentences, ["a", "b"])
    train_ids, test_ids = split_examples(examples, sentences)
    model = QCSEModel(["a", "b"])
    before = model.weights.copy()
    cfg = TrainConfig(epochs=2, learning_rate=0.02, batch_size=3)
    history, embeddings = train(model, examples, train_ids, test_ids, cfg)
    assert np.isfinite(embeddings).all()
    assert not np.allclose(before, model.weights)
    assert len(history) == 3
    second = QCSEModel(["a", "b"])
    history2, _ = train(second, examples, train_ids, test_ids, cfg)
    assert history == history2
    np.testing.assert_allclose(model.weights, second.weights)


def test_paper_accuracy_is_not_exact_accuracy():
    scores = metrics(np.array([[0.9, 0.1]]), np.array([[1, 1]]))
    assert scores["paper_similarity_accuracy"] == 1
    assert scores["exact_word_accuracy"] == 0
    assert scores["bit_accuracy"] == 0.5


def test_training_reuses_epoch_predictions_and_resumes(monkeypatch):
    sentences = [["a", "b", "a"], ["b", "a", "b"], ["a", "a", "b"]]
    examples = make_examples(sentences, ["a", "b"])
    train_ids, test_ids = split_examples(examples, sentences)
    model = QCSEModel(["a", "b"])
    predict = model.predict_encoded
    evaluated_sizes = []

    def counted_predict(states, weights=None):
        if weights is None:
            evaluated_sizes.append(len(states))
        return predict(states, weights)

    monkeypatch.setattr(model, "predict_encoded", counted_predict)
    checkpoints = []
    cfg = TrainConfig(epochs=1, batch_size=3)
    _, embeddings = train(
        model, examples, train_ids, test_ids, cfg, checkpoint_callback=checkpoints.append
    )
    assert evaluated_sizes == [len({e.context for e in examples})] * 2
    np.testing.assert_array_equal(embeddings, checkpoints[-1]["embeddings"])
    expected = predict([model.encode(e.context) for e in examples])
    np.testing.assert_allclose(embeddings, expected)
    history, resumed = train(
        model, examples, train_ids, test_ids, cfg, initial_state=checkpoints[-1]
    )
    uninterrupted = QCSEModel(["a", "b"])
    full_history, full_embeddings = train(
        uninterrupted, examples, train_ids, test_ids, TrainConfig(epochs=2, batch_size=3)
    )
    assert history == full_history
    np.testing.assert_array_equal(model.weights, uninterrupted.weights)
    np.testing.assert_array_equal(resumed, full_embeddings)


def test_cli_end_to_end(tmp_path, monkeypatch):
    data = tmp_path / "phrases.csv"
    data.write_text("a b c\nb c a\nc a b\na a c\n")
    out = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["qcse", "prepare", "--data", str(data), "--output", str(out)])
    main()
    prepared = next(out.iterdir())
    assert json.loads((prepared / "summary.json").read_text())["objective"] == "causal"
    with np.load(prepared / "contexts.npz") as archive:
        assert len(archive["target_ids"]) == 8
        assert archive["matrix_offsets"][-1] == len(archive["matrix_values"])
    monkeypatch.setattr(
        "sys.argv",
        [
            "qcse",
            "train",
            "--data",
            str(data),
            "--output",
            str(out),
            "--epochs",
            "1",
            "--max-examples",
            "6",
            "--objective",
            "cbow",
        ],
    )
    main()
    out = next(out.glob("train-*"))
    model = QCSEModel.load(out / "model.npz")
    assert len(model.embed_phrase("a b c")) == 3
    assert len(json.loads((out / "history.json").read_text())) == 2
    with np.load(out / "embeddings.npz") as archive:
        assert archive["probabilities"].shape == (6, 2)
        assert set(archive["train_ids"]).isdisjoint(archive["test_ids"])
    assert "OPENQASM 3" in (out / "circuit.qasm").read_text()

    monkeypatch.setattr("sys.argv", ["qcse", "continue", str(out / "run.npz"), "--epochs", "2"])
    main()
    with np.load(out / "run.npz", allow_pickle=False) as archive:
        history = json.loads(str(archive["history"]))
        assert history[-1]["epoch"] == 3
        assert len(history) == 4
        assert archive["embeddings"].shape == (6, 2)
