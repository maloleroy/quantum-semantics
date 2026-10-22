"""Exact differentiable evolution in the one-excitation vocabulary subspace.

These are classical statevector simulations, not hardware speedup demonstrations.
Dense spectral exponentiation is exact up to floating point and avoids Trotter error.
"""

import numpy as np
import torch
from torch import nn

from experiments.encoding_study.simulator import Simulator
from qcse.context import context_matrix, encoding_angles

from .data import graph_matrix, transition_probabilities


def context_distribution(contexts, size, decay=0.7, coherent=False):
    mask = contexts >= 0
    ages = torch.arange(contexts.shape[1] - 1, -1, -1, device=contexts.device)
    weights = decay**ages * mask
    if coherent:
        weights = weights.sqrt()
    values = torch.zeros((len(contexts), size), device=contexts.device)
    values.scatter_add_(1, contexts.clamp_min(0), weights)
    if coherent:
        values = values.square()
    return values / values.sum(1, keepdim=True)


class HamiltonianModel(nn.Module):
    hamiltonian: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    transition: torch.Tensor
    unigram: torch.Tensor
    stochastic: torch.Tensor

    def __init__(self, data, method="walk", layers=16, graph_kind="count", neighbors=24):
        super().__init__()
        self.method, self.layers = method, layers
        self.size = len(data["unigram"])
        h, w = graph_matrix(data["cooc"], graph_kind, neighbors)
        self.register_buffer("unigram", torch.tensor(data["unigram"], dtype=torch.float32))
        transition = transition_probabilities(data["transitions"], data["unigram"])
        self.register_buffer("transition", torch.tensor(transition, dtype=torch.float32))
        if method == "shuffled":
            perm = np.random.default_rng(723).permutation(self.size)
            h = h[np.ix_(perm, perm)]
        if method == "no_graph":
            h = np.zeros_like(h)
        self.directed = method == "directed"
        if self.directed:
            # Row=source, column=target. Small-time conditional Born law is exactly P.
            b = np.sqrt(transition)
            h = np.block([[np.zeros_like(b), b], [b.T, np.zeros_like(b)]])
            h /= np.linalg.norm(b, 2)
        eigenvalues, eigenvectors = np.linalg.eigh(h)
        self.register_buffer("eigenvalues", torch.tensor(eigenvalues, dtype=torch.float32))
        self.register_buffer("eigenvectors", torch.tensor(eigenvectors, dtype=torch.complex64))
        self.register_buffer("hamiltonian", torch.tensor(h, dtype=torch.float32))
        stochastic = np.divide(
            w, w.sum(1, keepdims=True), out=np.zeros_like(w), where=w.sum(1, keepdims=True) > 0
        )
        empty = w.sum(1) == 0
        stochastic[empty, empty] = 1
        self.register_buffer("stochastic", torch.tensor(stochastic, dtype=torch.float32))
        self.beta = nn.Parameter(0.6 + 0.15 * torch.randn(layers))
        self.gamma = nn.Parameter(0.15 * torch.randn(layers))
        self.mixture = nn.Parameter(torch.tensor(-2.2))
        if method in ("predictive_residual", "directed"):
            with torch.no_grad():
                self.beta.copy_(0.02 + 0.005 * torch.randn(layers))
                self.mixture.fill_(-6.0)

    def evolve(self, state, beta):
        q = self.eigenvectors
        return ((state @ q) * torch.exp(-1j * beta * self.eigenvalues)) @ q.T

    def forward(self, contexts, return_aux=False):
        coherent = self.method == "bag"
        decay = 1.0 if coherent else 0.7
        c = context_distribution(contexts, self.size, decay, coherent)
        if self.method in ("predictive", "predictive_residual"):
            initial = self.transition[contexts[:, -1]]
        elif self.directed:
            initial = torch.zeros_like(c).scatter_(1, contexts[:, -1:], 1.0)
        else:
            initial = c
        if self.method in ("diffusion", "directed_diffusion"):
            p = initial
            transition = self.transition if self.method == "directed_diffusion" else self.stochastic
            for b, g in zip(self.beta, self.gamma, strict=True):
                p = (1 - b.sigmoid()) * p + b.sigmoid() * (p @ transition)
                p = (1 - g.sigmoid()) * p + g.sigmoid() * c
            state = p.sqrt().to(torch.complex64)
            success = torch.ones(len(contexts), device=contexts.device)
        else:
            state = initial.sqrt().to(torch.complex64)
            potential = c
            if self.directed:
                state = torch.cat((state, torch.zeros_like(state)), dim=1)
                potential = torch.cat((c, c), dim=1)
            for layer, (b, g) in enumerate(zip(self.beta, self.gamma, strict=True)):
                # Context phase comes before graph mixing; a final phase is unobservable.
                if self.method != "static":
                    if self.method == "ordered":
                        slot = contexts[:, layer % contexts.shape[1]]
                        potential = torch.zeros_like(c).scatter_(
                            1, slot.clamp_min(0)[:, None], (slot >= 0).float()[:, None]
                        )
                    state = state * torch.exp(-1j * g * potential)
                if self.method != "no_graph":
                    state = self.evolve(state, b)
            born = state.abs().square()
            if self.directed:
                born = born[:, self.size :]
            success = born.sum(1)
            p = born / success[:, None].clamp_min(1e-20)
        mix = self.mixture.sigmoid()
        probabilities = (1 - mix) * p + mix * self.unigram
        if return_aux:
            return probabilities, {"state": state, "raw": p, "success": success, "mixture": mix}
        return probabilities


class ClassicalEmbedding(nn.Module):
    """Position-specific linear one-hot embedding and a deep residual MLP reference."""

    def __init__(self, size, window=8, layers=16, width=32):
        super().__init__()
        self.embedding = nn.Embedding(size + 1, width, padding_idx=size)
        self.position = nn.Parameter(torch.randn(window, width) * 0.1)
        self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.head = nn.Linear(width, size)
        self.size = size

    def forward(self, contexts, return_aux=False):
        mask = contexts >= 0
        tokens = torch.where(mask, contexts, self.size)
        x = (self.embedding(tokens) * self.position * mask[:, :, None]).sum(1)
        x = x / mask.sum(1, keepdim=True).sqrt()
        for layer in self.blocks:
            x = x + torch.tanh(layer(x)) / len(self.blocks)
        return self.head(x).softmax(1)


class QCSEModel(nn.Module):
    """Original fixed encoding and RX/RZ/CRZ ansatz, now with exact gradients."""

    bits: torch.Tensor
    z: torch.Tensor
    crz: torch.Tensor

    def __init__(self, size, layers=16, born_readout=False):
        super().__init__()
        self.size = size
        self.born_readout = born_readout
        self.qubits = (size - 1).bit_length()
        self.sim = Simulator(self.qubits)
        self.weights = nn.Parameter(
            torch.empty(layers, 3 * self.qubits - 1).uniform_(-np.pi, np.pi)
        )
        self.register_buffer("bits", torch.tensor(self.sim.bits, dtype=torch.float32))
        self.register_buffer("z", torch.tensor(self.sim.z, dtype=torch.float32))
        self.register_buffer("crz", torch.tensor(self.sim.crz, dtype=torch.float32))

    def encode(self, contexts):
        values = [
            encoding_angles(context_matrix(row[row >= 0].tolist(), self.size), self.qubits)
            for row in np.asarray(contexts)
        ]
        max_layers = max(len(a) for a in values)
        # Group by length: padded encoding layers would add unintended CNOT cascades.
        result = np.zeros((len(values), self.sim.dim), dtype=complex)
        for length in range(1, max_layers + 1):
            ids = [i for i, a in enumerate(values) if len(a) == length]
            if ids:
                result[ids] = self.sim.encode(np.array([values[i] for i in ids]), kind="paper")
        return torch.tensor(result, dtype=torch.complex64)

    def states(self, encoded):
        state = encoded
        for layer in self.weights:
            for q in range(self.qubits):
                v = state.reshape(len(state), -1, 2, 1 << q)
                a, b = v[:, :, 0], v[:, :, 1]
                c, s = torch.cos(layer[2 * q] / 2), torch.sin(layer[2 * q] / 2)
                state = torch.stack((c * a - 1j * s * b, c * b - 1j * s * a), dim=2).reshape(
                    len(state), -1
                )
            phase = self.z @ layer[1 : 2 * self.qubits : 2]
            if self.qubits > 1:
                phase = phase + layer[2 * self.qubits :] @ self.crz
            state = state * torch.exp(-0.5j * phase)
        return state

    def forward(self, encoded, return_aux=False):
        state = self.states(encoded)
        if self.born_readout:
            if self.size != self.sim.dim:
                raise ValueError("Joint Born control requires a power-of-two vocabulary")
            born = state.abs().square()
            p = born / born.sum(1, keepdim=True)
            return (p, {"state": state}) if return_aux else p
        bits = (state.abs().square() @ self.bits).clamp(1e-6, 1 - 1e-6)
        target_bits = self.bits[: self.size]
        scores = bits.log() @ target_bits.T + torch.log1p(-bits) @ (1 - target_bits).T
        p = scores.softmax(1)
        if return_aux:
            return p, {"bits": bits, "state": state}
        return p


class HamiltonianQCSEModel(QCSEModel):
    """Fixed graph encoding followed by the existing causal QCSE ansatz/readout.

    Layers=0 removes only the ansatz. The marginal-Z word decoder remains identical.
    Causality comes from the past-only contexts supplied by the dataset.
    """

    encoder_unitary: torch.Tensor

    def __init__(self, data, layers=16, evolution_time=8.0, graph_kind="count", neighbors=24):
        super().__init__(len(data["unigram"]), layers=layers)
        h, _ = graph_matrix(data["cooc"], graph_kind, neighbors)
        eigenvalues, eigenvectors = np.linalg.eigh(h)
        unitary = (eigenvectors * np.exp(-1j * evolution_time * eigenvalues)) @ eigenvectors.T
        self.register_buffer(
            "encoder_unitary", torch.tensor(unitary, dtype=torch.complex64), persistent=False
        )

    def encode(self, contexts):
        contexts = torch.as_tensor(contexts, dtype=torch.long)
        c = context_distribution(contexts, self.size)
        state = c.sqrt().to(torch.complex64) @ self.encoder_unitary.T
        if self.size < self.sim.dim:
            state = torch.cat(
                (state, state.new_zeros((len(state), self.sim.dim - self.size))), dim=1
            )
        return state
