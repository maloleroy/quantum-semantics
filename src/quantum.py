"""Differentiable CPU statevectors, small gates and exact Hermitian evolution."""

from typing import Literal, Protocol, cast

import torch
from torch import Tensor

Axis = Literal["x", "y", "z"]


class _LinearAlgebra(Protocol):
    def eigh(self, matrix: Tensor) -> tuple[Tensor, Tensor]: ...
    def qr(self, matrix: Tensor, mode: str = "reduced") -> tuple[Tensor, Tensor]: ...


# The public LAPACK binding lacks a complete Pyright signature in PyTorch.
_linalg = cast(_LinearAlgebra, torch.linalg)


def seed_random(seed: int) -> None:
    torch.random.set_rng_state(torch.Generator().manual_seed(seed).get_state())


def qubits(dimension: int) -> int:
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("State dimension must be a power of two >=2")
    return dimension.bit_length() - 1


def hermitian(coefficients: Tensor, dimension: int) -> Tensor:
    if coefficients.shape != (dimension**2,):
        raise ValueError("A d-dimensional Hermitian matrix needs d**2 real coordinates")
    rows, columns = torch.triu_indices(dimension, dimension, 1, device=coefficients.device)
    count = len(rows)
    entries = torch.complex(
        coefficients[dimension : dimension + count], coefficients[dimension + count :]
    )
    upper = torch.zeros((dimension, dimension), dtype=torch.complex128, device=coefficients.device)
    upper = upper.index_put((rows, columns), entries)
    return torch.diag(coefficients[:dimension]).to(torch.complex128) + upper + upper.mH


class _Context(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


class _Evolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Context, matrix: Tensor) -> Tensor:
        values, vectors = _linalg.eigh(matrix)
        ctx.save_for_backward(values, vectors)
        return (vectors * torch.exp(-1j * values)) @ vectors.mH

    @staticmethod
    def backward(ctx: _Context, *grad_outputs: Tensor) -> Tensor:
        gradient = grad_outputs[0]
        values, vectors = ctx.saved_tensors
        difference = values[:, None] - values[None, :]
        midpoint = (values[:, None] + values[None, :]) / 2
        # The sinc form has a finite derivative even at repeated eigenvalues.
        divided = -1j * torch.exp(-1j * midpoint) * torch.sinc(difference / (2 * torch.pi))
        result = vectors @ (divided.conj() * (vectors.mH @ gradient @ vectors)) @ vectors.mH
        return (result + result.mH) / 2


def evolve(matrix: Tensor) -> Tensor:
    """Exact exp(-iH); first-order derivatives only."""
    return cast(Tensor, _Evolution.apply(matrix))


def rotate(states: Tensor, qubit: int, angle: Tensor, axis: Axis) -> Tensor:
    count = qubits(states.shape[1])
    if not 0 <= qubit < count or angle.numel() not in (1, len(states)):
        raise ValueError("Rotation dimensions do not match the states")
    values = states.reshape(len(states), -1, 2, 1 << qubit)
    a, b = values[:, :, 0], values[:, :, 1]
    c, s = torch.cos(angle / 2).reshape(-1, 1, 1), torch.sin(angle / 2).reshape(-1, 1, 1)
    if axis == "x":
        left, right = c * a - 1j * s * b, c * b - 1j * s * a
    elif axis == "y":
        left, right = c * a - s * b, s * a + c * b
    else:
        left, right = (c - 1j * s) * a, (c + 1j * s) * b
    return torch.stack((left, right), dim=2).reshape_as(states)


def cnot(states: Tensor, control: int, target: int) -> Tensor:
    count = qubits(states.shape[1])
    if control == target or min(control, target) < 0 or max(control, target) >= count:
        raise ValueError("Invalid CNOT pair")
    indices = torch.arange(states.shape[1], device=states.device)
    permutation = indices ^ (((indices >> control) & 1) << target)
    return states[:, permutation]


def ansatz(states: Tensor, weights: Tensor) -> Tensor:
    count = qubits(states.shape[1])
    if weights.ndim != 2 or weights.shape[1] != 3 * count - 1:
        raise ValueError("Expected RX/RZ/CRZ weights for each layer")
    indices = torch.arange(states.shape[1], device=states.device)
    bits = ((indices[:, None] >> torch.arange(count, device=states.device)) & 1).double()
    z = 1 - 2 * bits
    for layer in weights:
        for q in range(count):
            states = rotate(states, q, layer[2 * q], "x")
        phase = z @ layer[1 : 2 * count : 2]
        for q in range(count - 1):
            phase = phase + layer[2 * count + q] * bits[:, q] * z[:, q + 1]
        states = states * torch.exp(-0.5j * phase)
    return states


def readout(states: Tensor, angles: Tensor) -> Tensor:
    if angles.shape != (qubits(states.shape[1]), 2):
        raise ValueError("Readout needs RX/RY angles for each measured qubit")
    for q, pair in enumerate(angles):
        states = rotate(states, q, pair[0], "x")
        states = rotate(states, q, pair[1], "y")
    return states


def independent_bits(probabilities: Tensor) -> Tensor:
    """Classical marginal decoder; not the circuit's joint one-shot distribution."""
    count = qubits(probabilities.shape[1])
    ids = torch.arange(probabilities.shape[1], device=probabilities.device)
    bits = ((ids[:, None] >> torch.arange(count, device=probabilities.device)) & 1).bool()
    marginals = probabilities @ bits.double()
    return torch.where(bits[None], marginals[:, None], 1 - marginals[:, None]).prod(-1)


def graph_hamiltonian(counts: Tensor, directed: bool = False) -> Tensor:
    """Training-count association graph, or Hermitian source/target dilation."""
    if directed:
        prior = (counts + 0.5) / (counts.sum(1, keepdim=True) + 0.5 * counts.shape[1])
        coupling = prior.sqrt()
        zero = torch.zeros_like(coupling)
        matrix = torch.cat((torch.cat((zero, coupling), 1), torch.cat((coupling.T, zero), 1)), 0)
        norm = _linalg.eigh(coupling.T @ coupling)[0][-1].sqrt()
        return (matrix / norm).to(torch.complex128)
    adjacency = counts + counts.T
    adjacency = adjacency - torch.diag(torch.diag(adjacency))
    degree = adjacency.sum(1).clamp_min(1).rsqrt()
    return (degree[:, None] * adjacency * degree[None]).to(torch.complex128)


def orthonormal_columns(factors: Tensor) -> Tensor:
    """Reduced complex QR for a full-column-rank spectral parameterization."""
    basis, triangular = _linalg.qr(torch.view_as_complex(factors.contiguous()))
    if torch.any(triangular.diagonal().abs() < 1e-10):
        raise ValueError("Rank-deficient Hamiltonian factors; QR gradients are undefined")
    return basis


def spectral_action(states: Tensor, factors: Tensor, energies: Tensor) -> Tensor:
    """Apply exp(-i Q diag(energies) Q†) without forming a dense operator."""
    basis = orthonormal_columns(factors)
    return states + ((states @ basis.conj()) * (torch.exp(-1j * energies) - 1)) @ basis.T
