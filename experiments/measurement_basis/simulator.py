"""Batched exact statevector simulator with local rotated readout."""

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
        self.crz = np.array(
            [self.bits[:, control] * self.z[:, target] for control, target in self.edges]
        )
        permutation = np.arange(self.dim)
        for control, target in self.edges:
            step = np.arange(self.dim)
            step ^= ((step >> control) & 1) << target
            permutation = permutation[step]
        self.cascade = permutation

    def initial(self, count, plus=False):
        if plus:
            return np.full((count, self.dim), 1 / np.sqrt(self.dim), dtype=complex)
        state = np.zeros((count, self.dim), dtype=complex)
        state[:, 0] = 1
        return state

    def rotate(self, state, qubit, angle, axis):
        values = state.reshape(len(state), -1, 2, 1 << qubit)
        cosine = np.cos(np.asarray(angle) / 2).reshape(-1, 1, 1)
        sine = np.sin(np.asarray(angle) / 2).reshape(-1, 1, 1)
        zero = values[:, :, 0, :].copy()
        one = values[:, :, 1, :].copy()
        if axis == "x":
            values[:, :, 0, :] = cosine * zero - 1j * sine * one
            values[:, :, 1, :] = -1j * sine * zero + cosine * one
        elif axis == "z":
            values[:, :, 0, :] = (cosine - 1j * sine) * zero
            values[:, :, 1, :] = (cosine + 1j * sine) * one
        else:
            raise ValueError(axis)

    def encode(self, angles):
        state = self.initial(len(angles), plus=True)
        for layer in range(angles.shape[1]):
            for q in range(self.qubits):
                self.rotate(state, q, angles[:, layer, q, 0], "x")
                self.rotate(state, q, angles[:, layer, q, 1], "z")
            state = state[:, self.cascade].copy()
        return state

    def ansatz(self, state, weights):
        state = state.copy()
        width = 3 * self.qubits - 1
        for layer in np.asarray(weights).reshape(-1, width):
            for q in range(self.qubits):
                self.rotate(state, q, layer[2 * q], "x")
            phase = self.z @ layer[1 : 2 * self.qubits : 2]
            if self.qubits > 1:
                phase += layer[2 * self.qubits :] @ self.crz
            state *= np.exp(-0.5j * phase)
        return state

    def probabilities(self, state):
        return (state.real**2 + state.imag**2) @ self.bits

    def bloch(self, state):
        """Return local Pauli X, Y and Z expectations for every qubit."""
        state = np.asarray(state)
        x = np.empty((len(state), self.qubits))
        y = np.empty_like(x)
        for q in range(self.qubits):
            values = state.reshape(len(state), -1, 2, 1 << q)
            coherence = np.sum(values[:, :, 0, :].conj() * values[:, :, 1, :], axis=(1, 2))
            x[:, q] = 2 * coherence.real
            y[:, q] = 2 * coherence.imag
        z = (state.real**2 + state.imag**2) @ self.z
        return np.stack((x, y, z), axis=-1)

    def readout_probabilities(self, state, basis="z", angles=None):
        """Measure after local learned rotations, without materializing rotated states."""
        if basis == "z":
            if angles is not None and len(angles):
                raise ValueError("Z readout has no angles")
            return self.probabilities(state)
        bloch = self.bloch(state)
        x, y, z = (bloch[:, :, index] for index in range(3))
        angles = np.asarray(angles, dtype=float)
        if basis == "learned_y":
            if angles.shape != (self.qubits,):
                raise ValueError("learned_y needs one angle per qubit")
            measured_z = -x * np.sin(angles) + z * np.cos(angles)
        elif basis == "learned_xyz":
            if angles.shape != (2 * self.qubits,):
                raise ValueError("learned_xyz needs two angles per qubit")
            rx, ry = angles.reshape(self.qubits, 2).T
            measured_z = -x * np.sin(ry) + (y * np.sin(rx) + z * np.cos(rx)) * np.cos(ry)
        else:
            raise ValueError(basis)
        return np.clip((1 - measured_z) / 2, 0, 1)

    def readout_gradient(self, state, targets, basis, angles):
        """Exact mean-BCE gradient for the local readout angles."""
        if basis == "z":
            return np.empty(0)
        bloch = self.bloch(state)
        x, y, z = (bloch[:, :, index] for index in range(3))
        angles = np.asarray(angles)
        if basis == "learned_y":
            measured_z = -x * np.sin(angles) + z * np.cos(angles)
            probabilities = np.clip((1 - measured_z) / 2, 1e-10, 1 - 1e-10)
            derivative = 0.5 * (x * np.cos(angles) + z * np.sin(angles))
            loss_derivative = (probabilities - targets) / (
                probabilities * (1 - probabilities) * probabilities.size
            )
            return np.sum(loss_derivative * derivative, axis=0)
        rx, ry = angles.reshape(self.qubits, 2).T
        common = y * np.sin(rx) + z * np.cos(rx)
        probabilities = np.clip(
            (1 + x * np.sin(ry) - common * np.cos(ry)) / 2, 1e-10, 1 - 1e-10
        )
        loss_derivative = (probabilities - targets) / (
            probabilities * (1 - probabilities) * probabilities.size
        )
        derivative_rx = -0.5 * (y * np.cos(rx) - z * np.sin(rx)) * np.cos(ry)
        derivative_ry = 0.5 * (x * np.cos(ry) + common * np.sin(ry))
        return np.stack(
            (
                np.sum(loss_derivative * derivative_rx, axis=0),
                np.sum(loss_derivative * derivative_ry, axis=0),
            ),
            axis=1,
        ).ravel()
