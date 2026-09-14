"""Single-head quantum attention with QCSE or learned amplitude encoding.

Simulates the postselected bilinear attention action from GPT on a Quantum
Computer, section 3.2. No softmax or classical Q/K/V projections are used.
Complex states use the real overlap (the Hadamard-test observable) as a score.
"""

import math
from typing import TypedDict

import numpy as np
import torch
from qiskit import QuantumCircuit
from torch import nn
from torch.nn import functional as F


class AttentionConfig(TypedDict):
    vocabulary_size: int
    window: int
    embedding_dim: int
    qubits: int
    layers: int
    alpha: float
    encoding: str
    context_alpha: float
    circuit: bool


class QuantumRegister(nn.Module):
    """Differentiable little-endian statevector gates using real/imaginary pairs."""

    z: torch.Tensor
    flips: torch.Tensor
    cnots: torch.Tensor

    def __init__(self, qubits):
        super().__init__()
        self.qubits = qubits
        indices = torch.arange(2**qubits)
        bits = (indices[:, None] >> torch.arange(qubits)) & 1
        self.register_buffer("z", (1 - 2 * bits).float())
        self.register_buffer("flips", indices[None] ^ (1 << torch.arange(qubits))[:, None])
        self.register_buffer(
            "cnots", indices[None] ^ (bits[:, :-1].T << torch.arange(1, qubits)[:, None])
        )

    def rotate(self, state, angle, qubit, axis):
        c = torch.cos(angle / 2).reshape(-1, 1)
        s = torch.sin(angle / 2).reshape(-1, 1)
        real, imag = state.unbind(-1)
        fr, fi = state[:, self.flips[qubit]].unbind(-1)
        if axis == "x":
            return torch.stack((c * real + s * fi, c * imag - s * fr), -1)
        sign = -self.z[:, qubit]
        return torch.stack((c * real + s * sign * fr, c * imag + s * sign * fi), -1)

    def phase(self, state, angles):
        phase = -0.5 * (angles @ self.z.T)
        c, s = torch.cos(phase), torch.sin(phase)
        real, imag = state.unbind(-1)
        return torch.stack((c * real - s * imag, s * real + c * imag), -1)

    def entangle(self, state):
        for permutation in self.cnots:
            state = state[:, permutation]
        return state

    def unitary(self, state, weights):
        for layer in weights:
            for q in range(self.qubits):
                state = self.rotate(state, layer[q, 0], q, "y")
            state = self.entangle(self.phase(state, layer[:, 1]))
        return state

    def features(self, state):
        real, imag = state.unbind(-1)
        fr, fi = state[:, self.flips].unbind(-1)
        x = (real[:, None] * fr + imag[:, None] * fi).sum(-1)
        y = ((real[:, None] * fi - imag[:, None] * fr) * self.z.T).sum(-1)
        z = (real.square() + imag.square()) @ self.z
        return torch.stack((x, y, z), -1).flatten(1)


class QuantumAttentionModel(nn.Module):
    positions: torch.Tensor

    def __init__(
        self,
        vocabulary_size,
        window=4,
        embedding_dim=16,
        qubits=4,
        layers=2,
        alpha=0.05,
        encoding="qcse",
        context_alpha=1.0,
        circuit=True,
    ):
        super().__init__()
        if min(vocabulary_size, window, embedding_dim, qubits, layers) < 1:
            raise ValueError("Model dimensions must be positive")
        if not all(math.isfinite(x) and x > 0 for x in (alpha, context_alpha)):
            raise ValueError("alpha and context_alpha must be positive and finite")
        if encoding not in ("qcse", "classical"):
            raise ValueError("encoding must be qcse or classical")
        if not isinstance(circuit, bool):
            raise ValueError("circuit must be a boolean")
        if qubits > 14 or embedding_dim > 2**qubits:
            raise ValueError("Use at most 14 qubits and embedding_dim <= 2**qubits")
        self.config: AttentionConfig = {
            "vocabulary_size": vocabulary_size,
            "window": window,
            "embedding_dim": embedding_dim,
            "qubits": qubits,
            "layers": layers,
            "alpha": alpha,
            "encoding": encoding,
            "context_alpha": context_alpha,
            "circuit": circuit,
        }
        self.register = QuantumRegister(qubits)
        self.qkv = nn.Parameter(alpha * torch.randn(3, layers, qubits, 2), requires_grad=circuit)
        self.decoder = nn.Linear(3 * qubits, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocabulary_size)
        self.embedding = (
            nn.Embedding(vocabulary_size + 1, embedding_dim, padding_idx=vocabulary_size)
            if encoding == "classical"
            else None
        )
        # Positions count real tokens, so left padding does not change their states.
        position = torch.arange(window)[:, None]
        frequency = torch.exp(-math.log(10000) * torch.arange(0, embedding_dim, 2) / embedding_dim)
        positions = torch.zeros(window, embedding_dim)
        positions[:, 0::2] = torch.sin(position * frequency)
        positions[:, 1::2] = torch.cos(position * frequency[: embedding_dim // 2])
        self.register_buffer("positions", positions)

    @torch.no_grad()
    def qcse_state(self, contexts):
        """Original exponential context matrix and H/RX/RZ/CNOT encoding."""
        qubits = self.config["qubits"]
        result = self.qkv.new_zeros(len(contexts), 2**qubits, 2)
        lengths = (contexts != self.config["vocabulary_size"]).sum(-1)
        for length in range(1, contexts.shape[1] + 1):
            selected = lengths == length
            ids = contexts[selected, -length:]
            if not len(ids):
                continue
            theta = ids.to(self.qkv.dtype) * (2 * math.pi / self.config["vocabulary_size"])
            positions = torch.arange(length, device=contexts.device)
            decay = torch.exp(
                -self.config["context_alpha"] * (positions[:, None] - positions[None]).abs()
            )
            matrix = decay * theta.sin()[:, :, None] * theta.cos()[:, None, :] + theta[:, :, None]
            angles = F.pad(matrix.flatten(1), (0, (-(length**2)) % (2 * qubits)))
            angles = angles.reshape(len(ids), -1, qubits, 2)
            state = self.qkv.new_zeros(len(ids), 2**qubits, 2)
            state[:, :, 0] = 2 ** (-qubits / 2)
            for layer in angles.unbind(1):
                for q in range(qubits):
                    state = self.register.rotate(state, layer[:, q, 0], q, "x")
                state = self.register.entangle(self.register.phase(state, layer[:, :, 1]))
            result[selected] = state
        return result

    def encode(self, contexts):
        valid = contexts != self.config["vocabulary_size"]
        if self.embedding is None:
            return torch.stack(
                [self.qcse_state(contexts[:, : i + 1]) for i in range(contexts.shape[1])], 1
            )
        positions = (valid.cumsum(-1) - 1).clamp_min(0)
        embedded = self.embedding(contexts) + self.positions[positions]
        amplitudes = F.normalize(embedded, dim=-1) * valid.unsqueeze(-1)
        amplitudes = F.pad(amplitudes, (0, 2 ** self.config["qubits"] - amplitudes.shape[-1]))
        return torch.stack((amplitudes, torch.zeros_like(amplitudes)), -1)

    def attend(self, states, valid):
        """Apply masked real overlaps to quantum values, then postselect/normalize.

        For row i, y_i = sum_{j<=i} Re(<q_i|k_j>) |v_j>. Dividing the
        masked overlap matrix by sequence length gives a contraction suitable
        for block encoding. Its scale cancels when the output is normalized.
        """
        batch, length, width, _ = states.shape
        if self.config["circuit"]:
            query, key, value = [
                self.register.unitary(states.reshape(-1, width, 2), weights).reshape(
                    batch, length, width * 2
                )
                for weights in self.qkv
            ]
        else:
            query = key = value = states.reshape(batch, length, width * 2)
        scores = query @ key.transpose(-1, -2)
        causal = torch.ones(length, length, dtype=torch.bool, device=states.device).tril()
        allowed = causal & valid[:, None, :] & valid[:, :, None]
        scores = scores.masked_fill(~allowed, 0)
        mixed = scores @ value
        norm = torch.linalg.vector_norm(mixed, dim=-1, keepdim=True)
        attended = (mixed / norm.clamp_min(1e-12)).reshape(batch, length, width, 2)
        return attended, scores

    def state(self, contexts):
        valid = contexts != self.config["vocabulary_size"]
        return self.attend(self.encode(contexts), valid)[0][:, -1]

    def representation(self, contexts):
        return self.decoder(self.register.features(self.state(contexts)))

    def scores(self, contexts, similarity="dot"):
        hidden = self.representation(contexts)
        if similarity == "dot":
            return self.output(hidden)
        if similarity == "cosine":
            return F.linear(F.normalize(hidden, dim=-1), F.normalize(self.output.weight, dim=-1))
        raise ValueError("similarity must be dot or cosine")

    def forward(self, contexts):
        return self.scores(contexts)

    @torch.no_grad()
    def attention_circuit(self, context):
        """Small reference circuit; ancilla=0 and last-position postselection.

        Dense state preparation and SVD block encoding verify the simulated
        action. They are not efficient oracle synthesis for quantum hardware.
        """
        contexts = torch.as_tensor(context, device=self.qkv.device).reshape(1, -1)
        valid = contexts != self.config["vocabulary_size"]
        if not bool(valid[0, -1]):
            raise ValueError("The last position must contain a token")
        states = self.encode(contexts)
        _, scores = self.attend(states, valid)
        length = contexts.shape[1]
        position_qubits = (length - 1).bit_length()
        positions = 2**position_qubits
        matrix = np.zeros((positions, positions))
        matrix[:length, :length] = scores[0].cpu().double().numpy() / length
        u, singular, vh = np.linalg.svd(matrix)
        complement = np.sqrt(np.maximum(0, 1 - singular**2))
        dilation = np.block(
            [
                [matrix, (u * complement) @ u.T],
                [(vh.T * complement) @ vh, -matrix.T],
            ]
        )
        qubits = self.config["qubits"]
        amplitudes = states[0].cpu().double().numpy()
        amplitudes = amplitudes[..., 0] + 1j * amplitudes[..., 1]
        initial = np.zeros((positions, 2**qubits), dtype=complex)
        initial[:length] = amplitudes
        initial /= np.linalg.norm(initial)
        circuit = QuantumCircuit(qubits + position_qubits + 1)
        circuit.initialize(initial.ravel().tolist(), range(qubits + position_qubits))
        for layer in self.qkv[2].cpu().double().numpy():
            for q in range(qubits):
                circuit.ry(float(layer[q, 0]), q)
                circuit.rz(float(layer[q, 1]), q)
            for q in range(qubits - 1):
                circuit.cx(q, q + 1)
        circuit.unitary(dilation, range(qubits, circuit.num_qubits), label="masked attention")
        return circuit

    @torch.no_grad()
    def entanglement(self, contexts):
        state = self.state(contexts).cpu().numpy()
        amplitudes = state[..., 0] + 1j * state[..., 1]
        cut = self.config["qubits"] // 2
        singular = np.linalg.svd(amplitudes.reshape(len(contexts), 2**cut, -1), compute_uv=False)
        probabilities = singular**2
        return {
            "mean_entropy_bits": float(
                -(probabilities * np.log2(np.clip(probabilities, 1e-12, 1))).sum(-1).mean()
            ),
            "max_schmidt_rank": int((singular > 1e-6).sum(-1).max()),
            "cut": cut,
        }
