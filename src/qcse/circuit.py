"""Qiskit circuits for Eqs. (10)-(18), without measurement collapse."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

DEFAULT_LAYERS = 2
MEASUREMENT_BASES = ("z", "learned_y", "learned_xyz")


def edges(qubits: int, direction: str):
    if direction not in ("forward", "reverse"):
        raise ValueError("direction must be forward (equations) or reverse (figures)")
    return [(q, q + 1) if direction == "forward" else (q + 1, q) for q in range(qubits - 1)]


def encoding_circuit(angles, direction: str = "forward") -> QuantumCircuit:
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 3 or angles.shape[2] != 2 or min(angles.shape) < 1:
        raise ValueError("angles must have shape (layers, qubits, 2)")
    qubits = angles.shape[1]
    circuit = QuantumCircuit(qubits, name="context")
    circuit.h(range(qubits))
    for layer in angles:
        for q, (rx, rz) in enumerate(layer):
            circuit.rx(float(rx), q)
            circuit.rz(float(rz), q)
        for control, target in edges(qubits, direction):
            circuit.cx(control, target)
    return circuit


def ansatz_circuit(qubits: int, layers: int = DEFAULT_LAYERS, direction: str = "forward"):
    if qubits < 1 or layers < 1:
        raise ValueError("qubits and layers must be positive")
    parameters = ParameterVector("theta", layers * (3 * qubits - 1))
    circuit = QuantumCircuit(qubits, name="ansatz")
    k = 0
    for _ in range(layers):
        for q in range(qubits):
            circuit.rx(parameters[k], q)
            circuit.rz(parameters[k + 1], q)
            k += 2
        for control, target in edges(qubits, direction):
            circuit.crz(parameters[k], control, target)
            k += 1
    return circuit, parameters


def measurement_circuit(qubits: int, basis: str = "z"):
    """Return a local readout rotation and its trainable parameters.

    Measurement remains computational-basis measurement after this circuit.  A
    single RY angle selects an axis in the X-Z plane.  RX followed by RY has two
    effective degrees of freedom and can select any single-qubit measurement
    axis; a third Euler angle would be redundant for Z measurement.
    """
    if qubits < 1:
        raise ValueError("qubits must be positive")
    if basis not in MEASUREMENT_BASES:
        raise ValueError(f"measurement basis must be one of {MEASUREMENT_BASES}")
    circuit = QuantumCircuit(qubits, name="readout")
    count = {"z": 0, "learned_y": qubits, "learned_xyz": 2 * qubits}[basis]
    parameters = ParameterVector("readout", count)
    if basis == "learned_y":
        for q in range(qubits):
            circuit.ry(parameters[q], q)
    elif basis == "learned_xyz":
        for q in range(qubits):
            circuit.rx(parameters[2 * q], q)
            circuit.ry(parameters[2 * q + 1], q)
    return circuit, parameters
