"""Batched statevector marginals with PyTorch on CPU, CUDA or Apple MPS."""

import numpy as np
import torch

from .circuit import MEASUREMENT_BASES, edges


class TensorSimulator:
    def __init__(
        self, qubits, layers, direction, device="cpu", batch_size=256, measurement_basis="z"
    ):
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be cpu, cuda or mps")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable; install a CUDA-enabled PyTorch or use --device cpu"
            )
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Apple MPS is unavailable; use --device cpu")
        if batch_size < 1:
            raise ValueError("simulation_batch_size must be positive")
        if measurement_basis not in MEASUREMENT_BASES:
            raise ValueError(f"measurement basis must be one of {MEASUREMENT_BASES}")
        self.qubits, self.layers, self.batch_size = qubits, layers, batch_size
        self.measurement_basis = measurement_basis
        self.device = torch.device(device)
        self.dtype = torch.float64 if device == "cpu" else torch.float32
        bits = ((np.arange(2**qubits)[:, None] >> np.arange(qubits)) & 1).astype(float)
        self.bits = self.tensor(bits)
        # RZ and CRZ commute: combine all diagonal gates into one phase per layer.
        z = 1 - 2 * bits
        self.z = self.tensor(z)
        self.phase_signs = np.column_stack(
            [z] + [bits[:, control] * z[:, target] for control, target in edges(qubits, direction)]
        )
        indices = np.arange(2**qubits)
        self.flips = torch.as_tensor(
            indices[None] ^ (1 << np.arange(qubits))[:, None], device=self.device
        )
        self.cnots = tuple(
            torch.as_tensor(indices ^ (bits[:, control].astype(int) << target), device=self.device)
            for control, target in edges(qubits, direction)
        )

    def tensor(self, values):
        return torch.as_tensor(values, dtype=self.dtype, device=self.device)

    def prepare_states(self, states):
        """Pack Qiskit states once; training keeps this cache on the selected device."""
        if isinstance(states, torch.Tensor):
            packed = states.to(device=self.device, dtype=self.dtype)
        else:
            amplitudes = (
                np.asarray([state.data for state in states], dtype=np.complex128)
                if len(states)
                else np.empty((0, 2**self.qubits), dtype=np.complex128)
            )
            if amplitudes.shape != (len(states), 2**self.qubits):
                raise ValueError("Encoded states do not match the model qubit count")
            # Real/imaginary pairs also work on MPS without complex tensor kernels.
            packed = self.tensor(np.stack((amplitudes.real, amplitudes.imag), axis=-1))
        if packed.ndim != 3 or packed.shape[1:] != (2**self.qubits, 2):
            raise ValueError("Encoded states do not match the model qubit count")
        return packed

    @torch.inference_mode()
    def encode(self, angles):
        """Batch the fixed H/RX/RZ/CNOT context encoder on the selected device."""
        angles = np.asarray(angles, dtype=float)
        if angles.ndim != 4 or angles.shape[3] != 2 or not angles.shape[0] or not angles.shape[1]:
            raise ValueError("Angles must have shape (batch, layers, qubits, 2)")
        if angles.shape[2] != self.qubits or not np.isfinite(angles).all():
            raise ValueError("Angles do not match the simulator")
        values = self.tensor(angles)
        state = self.tensor(np.zeros((len(angles), 2**self.qubits, 2), dtype=float))
        state[..., 0] = 2 ** (-self.qubits / 2)
        for layer in range(angles.shape[1]):
            for qubit in range(self.qubits):
                rx = values[:, layer, qubit, 0]
                c = torch.cos(rx / 2).reshape(-1, 1)
                s = torch.sin(rx / 2).reshape(-1, 1)
                real, imag = state.unbind(-1)
                fr, fi = state[:, self.flips[qubit]].unbind(-1)
                state = torch.stack((c * real + s * fi, c * imag - s * fr), dim=-1)
                rz = values[:, layer, qubit, 1]
                phase = -0.5 * rz[:, None] * self.z[None, :, qubit]
                c, s = torch.cos(phase), torch.sin(phase)
                real, imag = state.unbind(-1)
                state = torch.stack((real * c - imag * s, real * s + imag * c), dim=-1)
            for permutation in self.cnots:
                state = state[:, permutation]
        return state

    @torch.inference_mode()
    def predict(self, states, weights):
        """Batch contexts and optionally independent weight vectors (e.g. SPSA +/-)."""
        weights = np.asarray(weights, dtype=float)
        single = weights.ndim == 1
        readout_count = {"z": 0, "learned_y": self.qubits, "learned_xyz": 2 * self.qubits}[
            self.measurement_basis
        ]
        ansatz_count = self.layers * (3 * self.qubits - 1)
        if (
            weights.ndim not in (1, 2)
            or weights.shape[-1] != ansatz_count + readout_count
            or not np.isfinite(weights).all()
            or weights.size == 0
        ):
            raise ValueError("Invalid model weight vector")
        values = weights.reshape(-1, ansatz_count + readout_count)
        parameters = values[:, :ansatz_count].reshape(-1, self.layers, 3 * self.qubits - 1)
        rotations = parameters[:, :, : 2 * self.qubits].reshape(-1, self.layers, self.qubits, 2)
        # RX acts on [Re(a0), Im(a0), Re(a1), Im(a1)] for each amplitude pair.
        rx_cosines = self.tensor(np.cos(rotations[..., 0] / 2))
        rx_sines = self.tensor(np.sin(rotations[..., 0] / 2))
        diagonal = np.concatenate((rotations[..., 1], parameters[:, :, 2 * self.qubits :]), axis=-1)
        # Final diagonal gates matter when the learned readout rotates the basis.
        phase_layers = self.layers if readout_count else self.layers - 1
        phase = -0.5 * (diagonal[:, :phase_layers] @ self.phase_signs.T)
        phases = self.tensor(np.stack((np.cos(phase), np.sin(phase)), axis=-1))
        count = len(parameters)
        probabilities = np.empty((count, len(states), self.qubits))
        for start in range(0, len(states), self.batch_size):
            encoded = self.prepare_states(states[start : start + self.batch_size])
            batch = encoded.unsqueeze(0).expand(count, -1, -1, -1)
            for layer in range(self.layers):
                for q in range(self.qubits):
                    # Qiskit is little endian: q0 pairs neighboring amplitudes.
                    c = rx_cosines[:, layer, q, None, None]
                    s = rx_sines[:, layer, q, None, None]
                    real, imag = batch.unbind(-1)
                    fr, fi = batch[:, :, self.flips[q]].unbind(-1)
                    batch = torch.stack((c * real + s * fi, c * imag - s * fr), dim=-1)
                if layer < phase_layers:
                    real, imag = batch[..., 0], batch[..., 1]
                    cosine = phases[:, layer, None, :, 0]
                    sine = phases[:, layer, None, :, 1]
                    batch = torch.stack(
                        (real * cosine - imag * sine, real * sine + imag * cosine), dim=-1
                    )
            if readout_count:
                readout = values[:, ansatz_count:].reshape(
                    count, self.qubits, 1 if self.measurement_basis == "learned_y" else 2
                )
                for q in range(self.qubits):
                    if self.measurement_basis == "learned_xyz":
                        rx = self.tensor(readout[:, q, 0])
                        c = torch.cos(rx / 2)[:, None, None]
                        s = torch.sin(rx / 2)[:, None, None]
                        real, imag = batch.unbind(-1)
                        fr, fi = batch[:, :, self.flips[q]].unbind(-1)
                        batch = torch.stack((c * real + s * fi, c * imag - s * fr), dim=-1)
                    ry = self.tensor(readout[:, q, -1])
                    c = torch.cos(ry / 2)[:, None, None, None]
                    s = torch.sin(ry / 2)[:, None, None, None]
                    flipped = batch[:, :, self.flips[q]]
                    batch = c * batch - s * self.z[None, None, :, q, None] * flipped
            marginals = batch.square().sum(dim=-1) @ self.bits
            probabilities[:, start : start + len(encoded)] = marginals.cpu().numpy()
        return probabilities[0] if single else probabilities
