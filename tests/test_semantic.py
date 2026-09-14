import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from qcse.semantic import SemanticModel


def test_circuit_matches_qiskit_and_all_parts_receive_gradients():
    torch.manual_seed(7)
    model = SemanticModel(8, window=2, qubits=3, layers=2).double()
    contexts = torch.tensor([[8, 1], [2, 3], [4, 5]])
    angles = model.encode(contexts)
    states = model.state_from_angles(angles)
    features = model.features(states)
    for index, inputs in enumerate(angles.detach().numpy()):
        circuit = QuantumCircuit(3)
        for q in range(3):
            circuit.ry(inputs[q], q)
            circuit.rz(inputs[q + 3], q)
        for weights in model.ansatz.detach().numpy():
            for q in range(3):
                circuit.rx(weights[q], q)
                circuit.rz(weights[q + 3], q)
            for q in range(2):
                circuit.crz(weights[6 + q], q, q + 1)
        reference = Statevector.from_instruction(circuit)
        actual = states[index].detach().numpy()
        np.testing.assert_allclose(actual[:, 0] + 1j * actual[:, 1], reference.data, atol=1e-12)
        expected = []
        for q in range(3):
            for axis in "XYZ":
                label = ["I"] * 3
                label[2 - q] = axis
                expected.append(reference.expectation_value(Pauli("".join(label))).real)
        np.testing.assert_allclose(features[index].detach().numpy(), expected, atol=1e-12)
    torch.nn.functional.cross_entropy(model(contexts), torch.tensor([2, 4, 6])).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0
    assert model.embedding.weight.grad is not None
    assert torch.count_nonzero(model.embedding.weight.grad[-1]) == 0


def test_token_ids_only_address_embedding_rows():
    torch.manual_seed(11)
    model = SemanticModel(8, window=2)
    reordered = SemanticModel(**model.config)
    reordered.load_state_dict(model.state_dict())
    permutation = torch.randperm(8)
    mapping = torch.cat((permutation, torch.tensor([8])))
    with torch.no_grad():
        reordered.embedding.weight[mapping] = model.embedding.weight
        reordered.output_bias[permutation] = model.output_bias
    contexts = torch.tensor([[8, 2], [5, 1], [0, 7]])
    torch.testing.assert_close(model(contexts), reordered(mapping[contexts])[:, permutation])
