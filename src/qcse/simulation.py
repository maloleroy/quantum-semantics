"""Batched statevector marginals with PyTorch on CPU, CUDA or Apple MPS."""

import numpy as np
import torch

from .circuit import edges


class TensorSimulator:
    def __init__(self, qubits, layers, direction, device="cpu", batch_size=256):
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
        self.qubits, self.layers, self.batch_size = qubits, layers, batch_size
        self.device = torch.device(device)
        self.dtype = torch.float64 if device == "cpu" else torch.float32
        bits = ((np.arange(2**qubits)[:, None] >> np.arange(qubits)) & 1).astype(float)
        self.bits = self.tensor(bits)
        # RZ and CRZ commute: combine all diagonal gates into one phase per layer.
        z = 1 - 2 * bits
        self.phase_signs = np.column_stack(
            [z] + [bits[:, control] * z[:, target] for control, target in edges(qubits, direction)]
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
    def predict(self, states, weights):
        """Batch contexts and optionally independent weight vectors (e.g. SPSA +/-)."""
        weights = np.asarray(weights, dtype=float)
        single = weights.ndim == 1
        if (
            weights.ndim not in (1, 2)
            or weights.shape[-1] != self.layers * (3 * self.qubits - 1)
            or not np.isfinite(weights).all()
            or weights.size == 0
        ):
            raise ValueError("Invalid ansatz weight vector")
        parameters = weights.reshape(-1, self.layers, 3 * self.qubits - 1)
        rotations = parameters[:, :, : 2 * self.qubits].reshape(-1, self.layers, self.qubits, 2)
        # RX acts on [Re(a0), Im(a0), Re(a1), Im(a1)] for each amplitude pair.
        c, s = np.cos(rotations[..., 0] / 2), np.sin(rotations[..., 0] / 2)
        mixing = np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
        gates = self.tensor(c[..., None, None] * np.eye(4) + s[..., None, None] * mixing)
        diagonal = np.concatenate((rotations[..., 1], parameters[:, :, 2 * self.qubits :]), axis=-1)
        phase = -0.5 * (diagonal[:, :-1] @ self.phase_signs.T)
        phases = self.tensor(np.stack((np.cos(phase), np.sin(phase)), axis=-1))
        count = len(parameters)
        probabilities = np.empty((count, len(states), self.qubits))
        for start in range(0, len(states), self.batch_size):
            encoded = self.prepare_states(states[start : start + self.batch_size])
            batch = encoded.unsqueeze(0).expand(count, -1, -1, -1)
            for layer in range(self.layers):
                for q in range(self.qubits):
                    # Qiskit is little endian: q0 pairs neighboring amplitudes.
                    pairs = batch.reshape(count, -1, 2, 1 << q, 2).transpose(2, 3)
                    evolved = pairs.reshape(count, -1, 4) @ gates[:, layer, q].transpose(1, 2)
                    batch = (
                        evolved.reshape(count, -1, 1 << q, 2, 2)
                        .transpose(2, 3)
                        .reshape(batch.shape)
                    )
                # The final diagonal gates cannot change measured Z marginals.
                if layer < self.layers - 1:
                    real, imag = batch[..., 0], batch[..., 1]
                    cosine = phases[:, layer, None, :, 0]
                    sine = phases[:, layer, None, :, 1]
                    batch = torch.stack(
                        (real * cosine - imag * sine, real * sine + imag * cosine), dim=-1
                    )
            marginals = batch.square().sum(dim=-1) @ self.bits
            probabilities[:, start : start + len(encoded)] = marginals.cpu().numpy()
        return probabilities[0] if single else probabilities
