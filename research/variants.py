"""Frozen ordering variants and an exact batched evaluator of Qiskit gates."""

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qcse.context import context_matrix, encoding_angles
from qcse.model import QCSEModel


@dataclass(frozen=True)
class Variant:
    name: str = "canonical"
    input_ids: str = "canonical"  # canonical, fixed, per_layer
    map_seed: int = 101
    layout: str = "canonical"  # canonical, fixed_pairs, layer_pairs, reverse_layers, semantic
    rotations: str = "rx_rz"
    reverse_cnot: bool = False
    entangle: bool = True
    direct: bool = False

    def __post_init__(self):
        choices = {
            "input_ids": ("canonical", "fixed", "per_layer"),
            "layout": ("canonical", "fixed_pairs", "layer_pairs", "reverse_layers", "semantic"),
            "rotations": ("rx_rz", "rz_rx"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"Unknown {name}: {getattr(self, name)}")
        if self.layout == "semantic" and self.input_ids == "per_layer":
            raise ValueError("Semantic layout with layer-specific IDs is not implemented")


def reorder_rotations(circuit):
    """Reverse each adjacent RX/RZ pair while keeping angles on their axes."""
    result = QuantumCircuit(circuit.num_qubits)
    data = list(circuit.data)
    i = 0
    while i < len(data):
        item = data[i]
        if item.operation.name == "rx":
            following = data[i + 1]
            assert following.operation.name == "rz" and item.qubits == following.qubits
            result.append(following.operation, following.qubits)
            result.append(item.operation, item.qubits)
            i += 2
        else:
            result.append(item.operation, item.qubits)
            i += 1
    return result


def batch_evolve(states, circuit):
    """Apply actual bound RX/RZ/CRZ matrices to batched full complex states.

    Exact statevector simulation, no learned shortcut, truncation or shot sampling.
    Specialized to the reference ansatz. Qiskit remains the independent oracle.
    """
    values = np.asarray([state.data for state in states], dtype=complex).copy()
    basis = np.arange(values.shape[1])
    for item in circuit.data:
        gate = item.operation
        qubits = [circuit.find_bit(q).index for q in item.qubits]
        matrix = gate.to_matrix()
        if gate.name in ("rz", "crz"):
            local_ids = sum(((basis >> q) & 1) << j for j, q in enumerate(qubits))
            values *= np.diag(matrix)[local_ids]
        elif gate.name == "rx":
            q = qubits[0]
            zero = basis[(basis & (1 << q)) == 0]
            one = zero | (1 << q)
            a, b = values[:, zero].copy(), values[:, one].copy()
            values[:, zero] = matrix[0, 0] * a + matrix[0, 1] * b
            values[:, one] = matrix[1, 0] * a + matrix[1, 1] * b
        else:
            raise ValueError(f"Unsupported ansatz gate: {gate.name}")
    return values


class ResearchModel(QCSEModel):
    def __init__(self, vocabulary, variant=None, vectors=None, projected=None, **kwargs):
        super().__init__(vocabulary, **kwargs)
        self.variant = variant or Variant()
        self.vectors = vectors
        self.projected = projected
        v = self.variant
        if v.rotations == "rz_rx":
            self.ansatz = reorder_rotations(self.ansatz)
        if not v.entangle:
            # Keep parameter slots to pair initialization, but omit entangling gates.
            local = QuantumCircuit(self.qubits)
            for item in self.ansatz.data:
                if item.operation.name != "crz":
                    local.append(item.operation, item.qubits)
            self.ansatz = local

    def bound_ansatz(self, weights=None):
        values = self.weights if weights is None else np.asarray(weights)
        if values.shape != self.weights.shape or not np.isfinite(values).all():
            raise ValueError("Invalid weights")
        present = self.ansatz.parameters
        return self.ansatz.assign_parameters(
            {p: w for p, w in zip(self.parameters, values) if p in present}
        )

    def angles(self, context):
        v = self.variant
        ids = np.asarray(context)
        if v.direct:
            if self.projected is None:
                raise ValueError("Direct embeddings require a fitted projection")
            # Same mean projected context vector is given to the classical control.
            return self.projected[ids].mean(axis=0).reshape(1, self.qubits, 2)
        size = len(self.vocabulary)

        def mapped(layer):
            if v.input_ids == "canonical":
                return ids
            seed = v.map_seed + (layer if v.input_ids == "per_layer" else 0)
            return np.random.default_rng(seed).permutation(size)[ids]

        angles = encoding_angles(context_matrix(mapped(0), size, self.context), self.qubits)
        if v.input_ids == "per_layer":
            # Recompute C using the layer's frozen dictionary, take its matching chunk.
            for layer in range(1, len(angles)):
                angles[layer] = encoding_angles(
                    context_matrix(mapped(layer), size, self.context), self.qubits
                )[layer]
        if v.layout == "semantic":
            if self.vectors is None:
                raise ValueError("Semantic layout requires vocabulary vectors")
            c = context_matrix(mapped(0), size, self.context)
            # Reorder existing relations after C construction: positional distances stay intact.
            x = self.vectors[ids]
            similarity = x @ x.T
            order = [0]
            remaining = list(range(1, len(ids)))
            while remaining:
                nxt = max(remaining, key=lambda j: (similarity[order[-1], j], -j))
                order.append(nxt)
                remaining.remove(nxt)
            angles = encoding_angles(c[np.ix_(order, order)], self.qubits)
        elif v.layout in ("fixed_pairs", "layer_pairs"):
            for layer in range(len(angles)):
                seed = v.map_seed + (layer if v.layout == "layer_pairs" else 0)
                order = np.random.default_rng(seed).permutation(self.qubits)
                angles[layer] = angles[layer, order]
        elif v.layout == "reverse_layers":
            angles = angles[::-1].copy()
        return angles

    def encoding(self, context):
        v = self.variant
        circuit = QuantumCircuit(self.qubits)
        circuit.h(range(self.qubits))
        for layer in self.angles(context):
            for q, (rx, rz) in enumerate(layer):
                if v.rotations == "rx_rz":
                    circuit.rx(float(rx), q)
                    circuit.rz(float(rz), q)
                else:
                    circuit.rz(float(rz), q)
                    circuit.rx(float(rx), q)
            edge_order = list(range(self.qubits - 1))
            if v.reverse_cnot:
                edge_order.reverse()
            if v.entangle:
                for q in edge_order:
                    circuit.cx(q, q + 1)
        return circuit

    def encode(self, context):
        return Statevector.from_instruction(self.encoding(context))

    def circuit(self, context, measured=False):
        result = self.encoding(context).compose(self.bound_ansatz())
        if measured:
            result.measure_all()
        return result

    def predict_encoded(self, states, weights=None):
        # Limit working memory even if a later experiment supplies the whole corpus.
        result = []
        circuit = self.bound_ansatz(weights)
        for start in range(0, len(states), 64):
            values = batch_evolve(states[start : start + 64], circuit)
            result.append(np.abs(values) ** 2 @ self._basis_bits)
        return np.concatenate(result)

    def save(self, path):
        # A base checkpoint would silently lose experimental maps/layouts.
        raise NotImplementedError(
            "Save research configuration and weights with the experiment runner"
        )
