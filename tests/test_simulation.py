import numpy as np
import pytest
import torch
from qiskit.quantum_info import Statevector

from qcse.cli import main
from qcse.data import make_examples, split_examples
from qcse.model import QCSEModel
from qcse.training import TrainConfig, load_run, train

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")),
    pytest.param(
        "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="No MPS")
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("qubits,layers", [(1, 1), (2, 3), (5, 2)])
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_tensor_batches_match_qiskit(device, qubits, layers, direction):
    model = QCSEModel(
        [str(i) for i in range(2**qubits)],
        layers=layers,
        direction=direction,
        device=device,
        simulation_batch_size=2,
    )
    rng = np.random.default_rng(12)
    amplitudes = rng.normal(size=(5, 2**qubits)) + 1j * rng.normal(size=(5, 2**qubits))
    amplitudes /= np.linalg.norm(amplitudes, axis=1, keepdims=True)
    states = [Statevector(row) for row in amplitudes]
    # Distinct weights verify that parallel SPSA lanes stay independent.
    weights = np.stack((model.weights, model.weights + rng.normal(size=model.weights.shape)))
    expected = []
    for values in weights:
        ansatz = model.bound_ansatz(values)
        evolved = [state.evolve(ansatz) for state in states]
        expected.append([[state.probabilities([q])[1] for q in range(qubits)] for state in evolved])
    before = amplitudes.copy()
    packed = model.prepare_states(states)
    assert packed.device.type == device
    snapshot = packed.clone()
    actual = model.predict_encoded(packed, weights)
    tolerance = 2e-6 if device != "cpu" else 1e-12
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        model.predict_encoded(states), expected[0], atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(packed, snapshot, rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray([state.data for state in states]), before)


def test_empty_batch_and_invalid_weights():
    model = QCSEModel(["a", "b", "c"])
    assert model.predict_encoded([]).shape == (0, 2)
    assert model.predict_encoded([], np.stack((model.weights, model.weights))).shape == (2, 0, 2)
    for weights in ([0], np.full_like(model.weights, np.nan), np.zeros((1, 1, len(model.weights)))):
        with pytest.raises(ValueError, match="weight"):
            model.predict_encoded([], weights)
    with pytest.raises(ValueError, match="qubit"):
        model.predict_encoded([Statevector([1, 0])])


def test_simulation_settings_are_runtime_options(tmp_path):
    model = QCSEModel(["a", "b", "c"], simulation_batch_size=1)
    path = tmp_path / "model.npz"
    model.save(path)
    restored = QCSEModel.load(path, simulation_batch_size=3)
    assert restored.simulator.batch_size == 3
    np.testing.assert_allclose(
        model.predict_encoded([model.encode([0, 1])]),
        restored.predict_encoded([restored.encode([0, 1])]),
    )
    with pytest.raises(ValueError, match="positive"):
        QCSEModel(["a", "b"], simulation_batch_size=0)
    with pytest.raises(ValueError, match="device"):
        QCSEModel(["a", "b"], device="invalid")


def test_parallel_spsa_matches_sequential_evaluations(monkeypatch):
    vocabulary = ["a", "b", "c", "d"]
    sentences = [["a", "b", "c"], ["c", "a", "d"], ["d", "b", "a"]]
    examples = make_examples(sentences, vocabulary)
    train_ids, test_ids = split_examples(examples, sentences)
    parallel, sequential = QCSEModel(vocabulary), QCSEModel(vocabulary)
    predict = sequential.predict_encoded

    def sequential_predict(states, weights=None):
        if weights is not None and weights.ndim == 2:
            return np.stack([predict(states, values) for values in weights])
        return predict(states, weights)

    monkeypatch.setattr(sequential, "predict_encoded", sequential_predict)
    cfg = TrainConfig(epochs=2, batch_size=3)
    _, actual = train(parallel, examples, train_ids, test_ids, cfg)
    _, expected = train(sequential, examples, train_ids, test_ids, cfg)
    np.testing.assert_allclose(parallel.weights, sequential.weights, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_cli_tensor_training_and_resume(device, tmp_path, monkeypatch):
    data = tmp_path / "phrases.csv"
    data.write_text("a b c\nb c a\nc a b\na a c\n")
    output = tmp_path / "run"

    def invoke(*args):
        monkeypatch.setattr(
            "sys.argv", ["qcse", *args, "--device", device, "--simulation-batch-size", "2"]
        )
        main()

    invoke(
        "train", "--data", str(data), "--output", str(output), "--epochs", "1", "--batch-size", "3"
    )
    output = next(output.iterdir())
    invoke("continue", str(output / "run.npz"), "--epochs", "1")
    saved = load_run(output / "run.npz")
    assert saved["state"]["epoch"] == 2
    assert np.isfinite(saved["state"]["embeddings"]).all()
    model = QCSEModel.load(output / "model.npz")
    expected = model.predict_encoded([model.encode(e.context) for e in saved["examples"]])
    np.testing.assert_allclose(saved["state"]["embeddings"], expected, atol=2e-6, rtol=2e-6)
    invoke("embed", "--model", str(output / "model.npz"), "a b c")
