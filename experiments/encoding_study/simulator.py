"""Batched complex128 statevectors implementing QCSE's gates exactly.

RZ and CRZ within one ansatz layer commute with rotations on other qubits,
so their diagonal action can be fused. The last diagonal is retained for
state-level comparisons. Qiskit remains the independent reference in tests.
"""

import numpy as np


class Simulator:
    def __init__(self, qubits, direction="forward"):
        self.qubits = qubits
        self.dim = 1 << qubits
        self.bits = ((np.arange(self.dim)[:, None] >> np.arange(qubits)) & 1).astype(float)
        self.z = 1 - 2 * self.bits
        self.edges = [
            (q, q + 1) if direction == "forward" else (q + 1, q) for q in range(qubits - 1)
        ]
        self.crz = np.array([self.bits[:, c] * self.z[:, t] for c, t in self.edges])
        permutation = np.arange(self.dim)
        for c, t in self.edges:
            p = np.arange(self.dim)
            p ^= ((p >> c) & 1) << t
            permutation = permutation[p]
        self.cascade = permutation

    def initial(self, count, plus=False):
        if plus:
            return np.full((count, self.dim), 1 / np.sqrt(self.dim), dtype=complex)
        state = np.zeros((count, self.dim), dtype=complex)
        state[:, 0] = 1
        return state

    def rotate(self, state, qubit, angle, axis):
        v = state.reshape(len(state), -1, 2, 1 << qubit)
        c = np.cos(np.asarray(angle) / 2).reshape(-1, 1, 1)
        s = np.sin(np.asarray(angle) / 2).reshape(-1, 1, 1)
        a = v[:, :, 0, :].copy()
        b = v[:, :, 1, :].copy()
        if axis == "x":
            v[:, :, 0, :] = c * a - 1j * s * b
            v[:, :, 1, :] = -1j * s * a + c * b
        elif axis == "y":
            v[:, :, 0, :] = c * a - s * b
            v[:, :, 1, :] = s * a + c * b
        elif axis == "z":
            v[:, :, 0, :] = (c - 1j * s) * a
            v[:, :, 1, :] = (c + 1j * s) * b
        else:
            raise ValueError(axis)

    def encode(self, angles, kind="ry", entangle=True, state=None):
        # angles: examples, layers, qubits, 2
        angles = np.asarray(angles)
        if state is None:
            state = self.initial(len(angles), plus=kind in ("paper", "rzrx"))
        else:
            state = state.copy()
        axes = {"paper": ("x", "z"), "rzrx": ("z", "x"), "ry": ("y", "z")}[kind]
        for layer in range(angles.shape[1]):
            for q in range(self.qubits):
                for slot, axis in enumerate(axes):
                    self.rotate(state, q, angles[:, layer, q, slot], axis)
            if entangle:
                state = state[:, self.cascade].copy()
        return state

    def ansatz(self, state, weights, reupload=None):
        state = state.copy()
        width = 3 * self.qubits - 1
        layers = np.asarray(weights).reshape(-1, width)
        for index, layer in enumerate(layers):
            for q in range(self.qubits):
                self.rotate(state, q, layer[2 * q], "x")
            phase = self.z @ layer[1 : 2 * self.qubits : 2]
            if self.qubits > 1:
                phase += layer[2 * self.qubits :] @ self.crz
            state *= np.exp(-0.5j * phase)
            if reupload is not None and index < len(layers) - 1:
                state = self.encode(reupload, state=state)
        return state

    def probabilities(self, state):
        return (state.real**2 + state.imag**2) @ self.bits
