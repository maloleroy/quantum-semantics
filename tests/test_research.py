import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qcse.model import QCSEModel
from research.variants import ResearchModel, Variant, batch_evolve


@pytest.mark.parametrize(
    "variant",
    [
        Variant(),
        Variant(rotations="rz_rx"),
        Variant(reverse_cnot=True),
        Variant(entangle=False),
        Variant(layout="layer_pairs"),
        Variant(input_ids="per_layer"),
    ],
)
def test_batch_matches_full_qiskit(variant):
    model = ResearchModel([str(i) for i in range(9)], variant, window=8, layers=3)
    contexts = [[1, 3, 2, 8, 4, 1], [0, 6], [7, 4, 3]]
    states = [model.encode(c) for c in contexts]
    actual = batch_evolve(states, model.bound_ansatz())
    for i, c in enumerate(contexts):
        oracle = Statevector.from_instruction(model.circuit(c))
        np.testing.assert_allclose(actual[i], oracle.data, atol=2e-14)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=2e-14)
    expected = np.array([s.evolve(model.bound_ansatz()).probabilities() for s in states])
    np.testing.assert_allclose(
        model.predict_encoded(states), expected @ model._basis_bits, atol=2e-14
    )


def test_canonical_extension_preserves_reference():
    vocab = [str(i) for i in range(19)]
    base = QCSEModel(vocab, window=8)
    extension = ResearchModel(vocab, window=8)
    context = [1, 3, 5, 7, 9, 11, 13, 17]
    np.testing.assert_allclose(base.encode(context).data, extension.encode(context).data)
    np.testing.assert_allclose(
        base.predict_encoded([base.encode(context)]),
        extension.predict_encoded([extension.encode(context)]),
        atol=1e-14,
    )


def test_layout_preserves_values_and_frozen_mapping():
    vocab = [str(i) for i in range(17)]
    context = [2, 4, 6, 8, 10, 12, 14, 16]
    reference = ResearchModel(vocab).angles(context)
    vectors = np.random.default_rng(7).normal(size=(17, 6))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    for layout in ["fixed_pairs", "layer_pairs", "reverse_layers", "semantic"]:
        model = ResearchModel(vocab, Variant(layout=layout), vectors=vectors)
        np.testing.assert_allclose(
            np.sort(model.angles(context).ravel()), np.sort(reference.ravel())
        )
        np.testing.assert_equal(model.angles(context), model.angles(context))
    fixed = ResearchModel(vocab, Variant(input_ids="fixed"))
    changing = ResearchModel(vocab, Variant(input_ids="per_layer"))
    np.testing.assert_equal(fixed.angles(context)[0], changing.angles(context)[0])
    assert not np.allclose(fixed.angles(context)[1:], changing.angles(context)[1:])


def test_inactive_rotations_and_final_diagonal_parameters():
    model = ResearchModel([str(i) for i in range(8)], layers=2)
    states = [model.encode([1, 2, 3, 4])]
    before = model.predict_encoded(states)
    # Final layer RZ and CRZ positions; only RX precedes diagonal measurements.
    offset = 3 * model.qubits - 1
    inactive = [offset + 2 * q + 1 for q in range(model.qubits)]
    inactive += list(range(offset + 2 * model.qubits, len(model.weights)))
    model.weights[inactive] += 0.7
    np.testing.assert_allclose(model.predict_encoded(states), before, atol=1e-14)
    original = model.encoding([1, 2, 3, 4])
    altered = original.copy()
    # Replace the first RX; whole state must agree up to global phase.
    from qiskit.circuit.library import RXGate

    first = next(i for i, x in enumerate(altered.data) if x.operation.name == "rx")
    item = altered.data[first]
    altered.data[first] = item.replace(operation=RXGate(float(item.operation.params[0]) + 0.8))
    assert Statevector.from_instruction(original).equiv(Statevector.from_instruction(altered))


def test_semantic_codes_unique_and_projection_train_only():
    from qcse.data import Example
    from research.semantics import fit_projection, semantic_codes

    x = np.random.default_rng(5).normal(size=(19, 12))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    codes = semantic_codes(x, 5)
    assert len(np.unique(codes)) == 19
    assert np.all((codes >= 0) & (codes < 32))
    examples = [Example(i, 0, i, (i, i + 1)) for i in range(10)]
    train_ids = np.arange(7)
    a, _ = fit_projection(x, examples, train_ids, 4)
    changed = examples[:7] + [Example(9, 0, 2, (17, 18))] * 3
    b, _ = fit_projection(x, changed, train_ids, 4)
    np.testing.assert_allclose(a, b)


def test_classical_fit_does_not_use_held_out_targets():
    from research.semantics import fit_linear

    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 3))
    y = rng.integers(2, size=(20, 4))
    a, _ = fit_linear(x, y, np.arange(15), steps=5)
    y[15:] = 1 - y[15:]
    b, _ = fit_linear(x, y, np.arange(15), steps=5)
    np.testing.assert_equal(a, b)
