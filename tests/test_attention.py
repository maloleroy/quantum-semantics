import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qcse.attention import QuantumAttentionModel
from qcse.circuit import encoding_circuit
from qcse.context import ContextConfig, context_matrix, encoding_angles


def test_fixed_qcse_encoding_and_query_circuit_match_qiskit():
    torch.manual_seed(12)
    model = QuantumAttentionModel(9, qubits=3, embedding_dim=8).double()
    contexts = torch.tensor([[9, 9, 1, 2], [2, 5, 0, 3]])
    states = model.encode(contexts)
    for row, context in enumerate(contexts.tolist()):
        for position in range(4):
            prefix = [i for i in context[: position + 1] if i != 9]
            if not prefix:
                assert torch.count_nonzero(states[row, position]) == 0
                continue
            angles = encoding_angles(context_matrix(prefix, 9, ContextConfig()), 3)
            circuit = encoding_circuit(angles)
            reference = Statevector.from_instruction(circuit)
            actual = states[row, position].numpy()
            np.testing.assert_allclose(actual[:, 0] + 1j * actual[:, 1], reference.data, atol=1e-12)
    circuit = QuantumCircuit(3)
    for layer in model.qkv[0].detach().numpy():
        for q in range(3):
            circuit.ry(layer[q, 0], q)
            circuit.rz(layer[q, 1], q)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
    state = states[1, -1]
    actual = model.register.unitary(state[None], model.qkv[0])[0].detach().numpy()
    reference = Statevector(state[:, 0].numpy() + 1j * state[:, 1].numpy()).evolve(circuit)
    np.testing.assert_allclose(actual[:, 0] + 1j * actual[:, 1], reference.data, atol=1e-12)


@pytest.mark.parametrize("encoding", ["qcse", "classical"])
def test_attention_matches_postselected_block_encoding_and_trains(encoding):
    torch.manual_seed(7)
    model = QuantumAttentionModel(
        9, window=3, qubits=3, embedding_dim=8, encoding=encoding
    ).double()
    contexts = torch.tensor([[9, 1, 2], [4, 5, 6]])
    for context in contexts:
        # Qiskit evolves a real unitary dilation on position + ancilla registers.
        reference = Statevector.from_instruction(model.attention_circuit(context))
        selected = reference.data[2 * 8 : 3 * 8]
        selected /= np.linalg.norm(selected)
        state = model.state(context[None])[0].detach().numpy()
        np.testing.assert_allclose(state[:, 0] + 1j * state[:, 1], selected, atol=1e-12)
    torch.nn.functional.cross_entropy(model(contexts), torch.tensor([3, 7])).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm() > 0, name
    assert model.qkv.grad is not None
    assert all(group.norm() > 0 for group in model.qkv.grad)
    if model.embedding is not None:
        assert model.embedding.weight.grad is not None
        assert torch.count_nonzero(model.embedding.weight.grad[-1]) == 0


@pytest.mark.parametrize("encoding", ["qcse", "classical"])
def test_causality_padding_and_checkpoint_roundtrip(encoding):
    torch.manual_seed(4)
    model = QuantumAttentionModel(9, encoding=encoding)
    contexts = torch.tensor([[9, 1, 2, 3], [9, 1, 7, 8]])
    valid = contexts != 9
    states, scores = model.attend(model.encode(contexts), valid)
    torch.testing.assert_close(states[0, :2], states[1, :2])
    assert torch.count_nonzero(scores.triu(1)) == 0
    assert torch.count_nonzero(scores[:, :, 0]) == 0
    assert torch.isfinite(states).all()
    torch.testing.assert_close(model(contexts[:1]), model(contexts[:1, 1:]))
    empty = torch.full((1, 4), 9)
    assert torch.count_nonzero(model.state(empty)) == 0
    restored = QuantumAttentionModel(**model.config)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(contexts), restored(contexts))


def test_classical_no_circuit_ablation_has_no_circuit_gradients():
    torch.manual_seed(9)
    model = QuantumAttentionModel(
        9, qubits=3, embedding_dim=8, encoding="classical", circuit=False
    ).double()
    contexts = torch.tensor([[9, 1, 2, 3], [4, 5, 6, 7]])
    loss = torch.nn.functional.cross_entropy(model(contexts), torch.tensor([2, 5]))
    loss.backward()
    assert model.qkv.grad is None
    assert model.embedding is not None and model.embedding.weight.grad is not None
    assert torch.isfinite(model(contexts)).all()
