import numpy as np
import pytest
import torch
from qiskit.quantum_info import Statevector

from qcse.model import QCSEModel


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
        ),
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="No MPS")
        ),
    ],
)
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
