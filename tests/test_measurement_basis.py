import argparse

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments.measurement_basis.run import encode_all, prepare
from experiments.measurement_basis.simulator import Simulator
from qcse.circuit import ansatz_circuit, encoding_circuit
from qcse.data import DEFAULT_DATA
from qcse.training import binary_cross_entropy


@pytest.mark.parametrize("qubits", [2, 3, 10])
def test_measurement_study_statevector_against_qiskit(qubits):
    rng = np.random.default_rng(177)
    angles = rng.normal(size=(2, 2, qubits, 2))
    weights = rng.normal(size=2 * (3 * qubits - 1))
    sim = Simulator(qubits)
    encoded = sim.encode(angles)
    actual = sim.ansatz(encoded, weights)
    ansatz, parameters = ansatz_circuit(qubits, 2)
    bound = ansatz.assign_parameters(dict(zip(parameters, weights, strict=True)))
    for row in range(len(angles)):
        expected_encoded = Statevector.from_instruction(encoding_circuit(angles[row]))
        expected = expected_encoded.evolve(bound)
        np.testing.assert_allclose(encoded[row], expected_encoded.data, atol=1e-12)
        np.testing.assert_allclose(actual[row], expected.data, atol=1e-12)


@pytest.mark.parametrize("basis", ["learned_y", "learned_xyz"])
def test_readout_probabilities_against_qiskit(basis):
    rng = np.random.default_rng(714)
    sim = Simulator(3)
    raw = rng.normal(size=(4, sim.dim)) + 1j * rng.normal(size=(4, sim.dim))
    states = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    angles = rng.normal(size=3 if basis == "learned_y" else 6)
    actual = sim.readout_probabilities(states, basis, angles)
    for row, state in enumerate(states):
        circuit = QuantumCircuit(3)
        if basis == "learned_y":
            for q, angle in enumerate(angles):
                circuit.ry(angle, q)
        else:
            for q, (rx, ry) in enumerate(angles.reshape(3, 2)):
                circuit.rx(rx, q)
                circuit.ry(ry, q)
        expected = Statevector(state).evolve(circuit)
        np.testing.assert_allclose(
            actual[row], [expected.probabilities([q])[1] for q in range(3)], atol=1e-12
        )


@pytest.mark.parametrize("basis", ["learned_y", "learned_xyz"])
def test_exact_readout_gradient_against_central_difference(basis):
    rng = np.random.default_rng(926)
    sim = Simulator(3)
    raw = rng.normal(size=(5, sim.dim)) + 1j * rng.normal(size=(5, sim.dim))
    states = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    targets = rng.integers(0, 2, size=(5, 3))
    angles = rng.normal(size=3 if basis == "learned_y" else 6)
    actual = sim.readout_gradient(states, targets, basis, angles)
    expected = np.empty_like(actual)
    epsilon = 1e-6
    for index in range(len(angles)):
        shift = np.zeros_like(angles)
        shift[index] = epsilon
        plus = sim.readout_probabilities(states, basis, angles + shift)
        minus = sim.readout_probabilities(states, basis, angles - shift)
        expected[index] = (
            binary_cross_entropy(plus, targets) - binary_cross_entropy(minus, targets)
        ) / (2 * epsilon)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def test_window_eight_encoding_preserves_variable_layer_counts():
    args = argparse.Namespace(data=DEFAULT_DATA, window=8, train_limit=10, val_limit=10)
    _, vocabulary, examples, _, _, _ = prepare(args)
    sample = [
        examples[next(i for i, example in enumerate(examples) if len(example.context) == length)]
        for length in (1, 5, 8)
    ]
    states = encode_all(Simulator(10), sample, len(vocabulary))
    np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1, atol=1e-12)
    assert not np.allclose(states[0], states[1])
