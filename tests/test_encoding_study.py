import argparse

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments.encoding_study.encoders import METHODS, Features
from experiments.encoding_study.run import prepare, vocabulary_metrics
from experiments.encoding_study.simulator import Simulator
from qcse.circuit import ansatz_circuit, edges
from qcse.context import context_matrix, encoding_angles
from qcse.data import DEFAULT_DATA, Example


@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("kind", ["paper", "rzrx", "ry"])
@pytest.mark.parametrize("qubits", [2, 3, 10])
def test_batched_simulator_against_qiskit(direction, kind, qubits):
    rng = np.random.default_rng(7)
    angles = rng.normal(size=(3, 2, qubits, 2))
    sim = Simulator(qubits, direction)
    state = sim.encode(angles, kind)
    ansatz, parameters = ansatz_circuit(qubits, 2, direction)
    weights = rng.normal(size=len(parameters))
    actual = sim.ansatz(state, weights)
    for row in range(len(angles)):
        circuit = QuantumCircuit(qubits)
        if kind in ("paper", "rzrx"):
            circuit.h(range(qubits))
        for layer in angles[row]:
            for q, (a, b) in enumerate(layer):
                if kind == "paper":
                    circuit.rx(a, q)
                    circuit.rz(b, q)
                elif kind == "rzrx":
                    circuit.rz(a, q)
                    circuit.rx(b, q)
                else:
                    circuit.ry(a, q)
                    circuit.rz(b, q)
            for c, t in edges(qubits, direction):
                circuit.cx(c, t)
        np.testing.assert_allclose(
            state[row], Statevector.from_instruction(circuit).data, atol=1e-12
        )
        circuit.compose(
            ansatz.assign_parameters(dict(zip(parameters, weights, strict=True))), inplace=True
        )
        expected = Statevector.from_instruction(circuit)
        np.testing.assert_allclose(actual[row], expected.data, atol=1e-12)
        np.testing.assert_allclose(
            sim.probabilities(actual)[row],
            [expected.probabilities([q])[1] for q in range(qubits)],
            atol=1e-12,
        )


def test_reupload_against_qiskit():
    sim = Simulator(3)
    rng = np.random.default_rng(27)
    angles = rng.normal(size=(1, 1, 3, 2))
    weights = rng.normal(size=16)
    actual = sim.ansatz(sim.encode(angles), weights, reupload=angles)[0]
    circuit = QuantumCircuit(3)
    ansatz, parameters = ansatz_circuit(3, 1)
    for layer in range(2):
        for q, (a, b) in enumerate(angles[0, 0]):
            circuit.ry(a, q)
            circuit.rz(b, q)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.compose(
            ansatz.assign_parameters(
                dict(zip(parameters, weights[layer * 8 : (layer + 1) * 8], strict=True))
            ),
            inplace=True,
        )
    np.testing.assert_allclose(actual, Statevector.from_instruction(circuit).data, atol=1e-12)


def test_single_token_information_loss_and_repair():
    sim = Simulator(10)
    angles = np.array([encoding_angles(context_matrix([i], 729), 10) for i in [0, 10, 400]])
    original = sim.encode(angles, "paper")
    np.testing.assert_allclose(np.abs(original @ original.conj().T) ** 2, 1, atol=1e-12)
    fixed = sim.encode(angles, "ry")
    assert np.abs(np.vdot(fixed[0], fixed[-1])) ** 2 < 0.99


def test_split_has_no_sentence_overlap():
    args = argparse.Namespace(data=DEFAULT_DATA, train_limit=640, val_limit=320)
    _, examples, train, val, test = prepare(args)
    from qcse.data import load_phrases

    sentences = load_phrases()
    groups = [{tuple(sentences[examples[i].sentence]) for i in ids} for ids in [train, val, test]]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]


def test_fitted_features_do_not_use_held_out_targets():
    examples = [Example(0, 1, 1, (0,)), Example(1, 1, 2, (1,)), Example(2, 1, 3, (2,))]
    changed = examples[:2] + [Example(2, 1, 0, (2,))]
    a = Features(examples, [0, 1], 4, 2)
    b = Features(changed, [0, 1], 4, 2)
    for name in ["ppmi", "bigram", "bigram_bits", "bit_prior", "unigram"]:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_all_candidate_states_are_normalized_and_finite():
    args = argparse.Namespace(data=DEFAULT_DATA, train_limit=20, val_limit=10)
    vocabulary, examples, train, _, _ = prepare(args)
    examples = examples[:12]
    features = Features(examples, np.arange(8), len(vocabulary), 10)
    sim = Simulator(10)
    for method in METHODS:
        spec = features.build(method)
        state = spec["state"]
        if state is None:
            state = sim.encode(spec["angles"], spec["kind"], spec["entangle"])
        assert np.isfinite(state).all(), method
        np.testing.assert_allclose(np.linalg.norm(state, axis=1), 1, atol=1e-12, err_msg=method)


def test_explicit_id_slots_retain_every_position_and_token_zero_presence():
    contexts: list[tuple[int, ...]] = [(1, 2, 3, 4)]
    for position in range(4):
        changed = [1, 2, 3, 4]
        changed[position] = 400
        contexts.append(tuple(changed))
    contexts.extend([(0, 1), (1,)])
    examples = [Example(i, 4, 0, c) for i, c in enumerate(contexts)]
    f = Features(examples, [0], 729, 10)
    spec = f.build("slots_id")
    state = Simulator(10).encode(spec["angles"])
    assert all(abs(np.vdot(state[0], s)) ** 2 < 0.99 for s in state[1:5])
    assert abs(np.vdot(state[-1], state[-2])) ** 2 < 0.99


def test_vocab_scores_are_normalized_independent_bit_scores():
    bits = np.array([[0, 0], [1, 0], [0, 1.0]])
    p = np.array([[0.8, 0.7]])
    result = vocabulary_metrics(p, np.array([1]), bits)
    expected = 0.8 * 0.3 / (1 - 0.8 * 0.7)
    assert result["vocab_nll"] == pytest.approx(-np.log(expected))
    assert result["invalid_id_rate"] == 1
    assert result["vocab_top1"] == 1


def test_batched_backend_preserves_existing_training_updates():
    from qcse.model import QCSEModel
    from qcse.training import TrainConfig, train

    class BatchedModel(QCSEModel):
        def predict_encoded(self, states, weights=None):
            sim = Simulator(self.qubits, self.direction)
            weights = self.weights if weights is None else weights
            return sim.probabilities(sim.ansatz(np.array([s.data for s in states]), weights))

    original = QCSEModel(["a", "b", "c", "d"], seed=12)
    batched = BatchedModel(["a", "b", "c", "d"], seed=12)
    examples = [Example(i, 1, (i + 1) % 4, (i % 4,)) for i in range(6)]
    config = TrainConfig(epochs=2, batch_size=2, learning_rate=0.01, seed=25)
    a, ap = train(original, examples, [0, 1, 2, 3], [4, 5], config)
    b, bp = train(batched, examples, [0, 1, 2, 3], [4, 5], config)
    np.testing.assert_allclose(original.weights, batched.weights, atol=1e-12)
    np.testing.assert_allclose(ap, bp, atol=1e-12)
    assert a[-1]["test"]["bce"] == pytest.approx(b[-1]["test"]["bce"], abs=1e-12)


def test_bigram_angles_without_ansatz_equal_the_classical_bit_probabilities():
    examples = [Example(i, 1, (i + 1) % 4, (i % 4,)) for i in range(7)]
    f = Features(examples, [0, 1, 2, 3], 729, 10)
    spec = f.build("bigram_angles")
    sim = Simulator(10)
    state = sim.encode(spec["angles"], spec["kind"], spec["entangle"])
    np.testing.assert_allclose(
        sim.probabilities(state), f.bigram_bits[[e.context[-1] for e in examples]], atol=1e-12
    )
