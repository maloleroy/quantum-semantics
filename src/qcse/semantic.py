"""Small differentiable word decoder inspired by Halil's BBQC roadmap.

Token IDs only address embedding rows. The circuit uses a fixed latent chain,
not an inferred language graph. Exact real/imaginary statevectors keep autograd
available on CPU, CUDA and Apple MPS; this is not a matrix-product-state backend.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SemanticModel(nn.Module):
    z: torch.Tensor
    phase_signs: torch.Tensor
    flips: torch.Tensor

    def __init__(self, vocabulary_size, window=4, embedding_dim=16, qubits=4, layers=2):
        super().__init__()
        if min(vocabulary_size, window, embedding_dim, qubits, layers) < 1:
            raise ValueError("Model dimensions must be positive")
        if qubits > 12:
            raise ValueError("This exact-state prototype supports at most 12 qubits")
        self.config = dict(
            vocabulary_size=vocabulary_size,
            window=window,
            embedding_dim=embedding_dim,
            qubits=qubits,
            layers=layers,
        )
        self.embedding = nn.Embedding(
            vocabulary_size + 1, embedding_dim, padding_idx=vocabulary_size
        )
        nn.init.normal_(self.embedding.weight, std=0.1)
        with torch.no_grad():
            self.embedding.weight[-1].zero_()
        self.encoder = nn.Linear(window * embedding_dim, 2 * qubits)
        self.ansatz = nn.Parameter(0.05 * torch.randn(layers, 3 * qubits - 1))
        self.decoder = nn.Linear(3 * qubits, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocabulary_size))
        indices = torch.arange(2**qubits)
        bits = ((indices[:, None] >> torch.arange(qubits)) & 1).float()
        z = 1 - 2 * bits
        self.register_buffer("z", z)
        # Directed chain 0 -> 1 -> ...; each CRZ phase is control_bit * target_Z.
        self.register_buffer("phase_signs", torch.cat((z, bits[:, :-1] * z[:, 1:]), dim=1))
        self.register_buffer("flips", indices[None, :] ^ (1 << torch.arange(qubits))[:, None])

    def encode(self, contexts):
        embedded = self.embedding(contexts).flatten(1)
        return math.pi * torch.tanh(self.encoder(embedded))

    def rotate(self, state, angle, qubit, axis):
        c, s = torch.cos(angle / 2).reshape(-1, 1), torch.sin(angle / 2).reshape(-1, 1)
        flipped = state[:, self.flips[qubit]]
        real, imag = state.unbind(-1)
        fr, fi = flipped.unbind(-1)
        if axis == "x":
            return torch.stack((c * real + s * fi, c * imag - s * fr), dim=-1)
        # RY mixes amplitudes with -sin for |0> and +sin for |1>.
        sign = -self.z[:, qubit]
        return torch.stack((c * real + s * sign * fr, c * imag + s * sign * fi), dim=-1)

    @staticmethod
    def phase(state, angle):
        real, imag = state.unbind(-1)
        c, s = torch.cos(angle), torch.sin(angle)
        return torch.stack((real * c - imag * s, real * s + imag * c), dim=-1)

    def state_from_angles(self, angles):
        qubits = self.config["qubits"]
        state = angles.new_zeros(len(angles), 2**qubits, 2)
        state[:, 0, 0] = 1
        for q in range(qubits):
            state = self.rotate(state, angles[:, q], q, "y")
        state = self.phase(state, -0.5 * (angles[:, qubits:] @ self.z.T))
        for weights in self.ansatz:
            for q in range(qubits):
                state = self.rotate(state, weights[q], q, "x")
            # Retain the final phases: they affect X/Y even when Z is unchanged.
            state = self.phase(state, -0.5 * (weights[qubits:] @ self.phase_signs.T))
        return state

    def features(self, state):
        real, imag = state.unbind(-1)
        fr, fi = state[:, self.flips].unbind(-1)
        x = (real[:, None] * fr + imag[:, None] * fi).sum(-1)
        y = ((real[:, None] * fi - imag[:, None] * fr) * self.z.T).sum(-1)
        z = (real.square() + imag.square()) @ self.z
        return torch.stack((x, y, z), dim=-1).flatten(1)

    def forward(self, contexts):
        return self.scores(contexts)

    def representation(self, contexts):
        features = self.features(self.state_from_angles(self.encode(contexts)))
        return self.decoder(features)

    def scores(self, contexts, similarity="dot"):
        hidden = self.representation(contexts)
        embeddings = self.embedding.weight[:-1]
        if similarity == "cosine":
            hidden = F.normalize(hidden, dim=-1)
            embeddings = F.normalize(embeddings, dim=-1)
        elif similarity != "dot":
            raise ValueError("similarity must be dot or cosine")
        return F.linear(hidden, embeddings, self.output_bias if similarity == "dot" else None)

    def set_trainable(self, groups):
        """Restrict optimization to named groups for ablation runs."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if "embedding" in groups:
            self.embedding.weight.requires_grad_(True)
        if "encoder" in groups:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(True)
        if "ansatz" in groups:
            self.ansatz.requires_grad_(True)
        if "decoder" in groups:
            for parameter in self.decoder.parameters():
                parameter.requires_grad_(True)
            self.output_bias.requires_grad_(True)

    @torch.no_grad()
    def entanglement(self, contexts):
        state = self.state_from_angles(self.encode(contexts)).cpu().numpy()
        amplitudes = state[..., 0] + 1j * state[..., 1]
        left = 2 ** (self.config["qubits"] // 2)
        singular = np.linalg.svd(amplitudes.reshape(len(contexts), left, -1), compute_uv=False)
        p = singular**2
        entropy = -(p * np.log2(np.clip(p, 1e-12, 1))).sum(-1)
        return {
            "mean_entropy_bits": float(entropy.mean()),
            "max_schmidt_rank": int((singular > 1e-6).sum(-1).max()),
            "cut": self.config["qubits"] // 2,
        }
