"""Small Qiskit compilation reference for the proposed single-particle encoders."""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate, StatePreparation
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit.synthesis import MatrixExponential
from scipy.linalg import expm

from .data import write_json


def binary_circuit(hamiltonian, potential, state, betas, gammas):
    """Exact matrix-exponential synthesis reference; no efficient-hardware claim."""
    size = len(hamiltonian)
    qubits = (size - 1).bit_length()
    dim = 1 << qubits
    padded = np.zeros((dim, dim), dtype=complex)
    padded[:size, :size] = hamiltonian
    diagonal = np.zeros(dim)
    diagonal[:size] = potential
    initial = np.zeros(dim, dtype=complex)
    initial[:size] = state
    circuit = QuantumCircuit(qubits)
    circuit.append(StatePreparation(initial.tolist()), range(qubits))
    h = SparsePauliOp.from_operator(Operator(padded))
    c = SparsePauliOp.from_operator(Operator(np.diag(diagonal)))
    for beta, gamma in zip(betas, gammas, strict=True):
        circuit.append(
            PauliEvolutionGate(c, time=float(gamma), synthesis=MatrixExponential()), range(qubits)
        )
        circuit.append(
            PauliEvolutionGate(h, time=float(beta), synthesis=MatrixExponential()), range(qubits)
        )
    return circuit


def run(output):
    rng = np.random.default_rng(314)
    records = []
    for size in [4, 8, 16]:
        w = rng.random((size, size))
        h = (w + w.T) / 2
        np.fill_diagonal(h, 0)
        h /= np.linalg.norm(h, 2)
        potential = rng.random(size)
        potential /= potential.sum()
        state = np.sqrt(potential).astype(complex)
        beta, gamma = rng.normal(size=(2, 4))
        circuit = binary_circuit(h, potential, state, beta, gamma)
        expected = state.copy()
        for b, g in zip(beta, gamma, strict=True):
            expected = expm(-1j * b * h) @ (np.exp(-1j * g * potential) * expected)
        actual = Statevector.from_instruction(circuit).data
        error = float(np.max(np.abs(actual - expected)))
        compiled = transpile(
            circuit.decompose(gates_to_decompose=["PauliEvolution"]),
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=1,
            seed_transpiler=42,
        )
        fidelity = abs(np.vdot(Statevector.from_instruction(compiled).data, expected)) ** 2
        records.append(
            {
                "vertices": size,
                "layers": 4,
                "qubits": circuit.num_qubits,
                "pauli_terms": len(SparsePauliOp.from_operator(Operator(h)).simplify()),
                "exact_state_error": error,
                "compiled_infidelity": float(abs(1 - fidelity)),
                "compiled_depth": compiled.depth(),
                "compiled_counts": dict(compiled.count_ops()),
            }
        )
    write_json(output, records)
    return records


if __name__ == "__main__":
    print(run("experiments/hamiltonian_study/results/circuit_resources.json"))
