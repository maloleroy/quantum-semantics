"""Fixed word amplitudes and the original QCSE numerical-ID context encodings."""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .quantum import cnot, qubits, rotate

ContextMethod = Literal["exponential", "diagonal", "phase", "hash", "angular"]


def amplitudes(
    contexts: Tensor, size: int, decay: float = 0.7, position_phase: float = math.pi / 2
) -> Tensor:
    if contexts.ndim != 2 or torch.any(contexts < -1) or torch.any(contexts >= size):
        raise ValueError("Expected word IDs and -1 padding")
    if not 0 < decay <= 1:
        raise ValueError("Decay must lie in (0,1]")
    ages = torch.arange(contexts.shape[1] - 1, -1, -1, dtype=torch.float64, device=contexts.device)
    weights = (decay**ages).sqrt() * torch.exp(1j * position_phase * ages)
    state = torch.zeros((len(contexts), size), dtype=torch.complex128, device=contexts.device)
    state.scatter_add_(1, contexts.clamp_min(0), (contexts >= 0) * weights)
    norm = state.abs().square().sum(dim=1, keepdim=True).sqrt()
    if torch.any(norm < 1e-12):
        raise ValueError("Empty or cancelled context")
    return state / norm


@dataclass(frozen=True)
class ContextConfig:
    method: ContextMethod = "exponential"
    alpha: float = 1.0
    omega: float = 1.0
    delta: float = 1.0
    prime: int = 31
    hash_size: int = 997


def context_matrices(ids: Tensor, size: int, config: ContextConfig) -> Tensor:
    if ids.ndim != 2 or not ids.numel() or torch.any(ids < 0) or torch.any(ids >= size):
        raise ValueError("Context matrices need nonempty, unpadded word IDs")
    if min(config.alpha, config.omega, config.delta, config.hash_size) <= 0:
        raise ValueError("Context constants must be positive")
    theta = ids.double() * (2 * math.pi / size)
    positions = torch.arange(ids.shape[1], dtype=torch.float64, device=ids.device)
    decay = torch.exp(-config.alpha * (positions[:, None] - positions[None]).abs())
    if config.method == "angular":
        side = math.ceil(math.sqrt(ids.shape[1]))
        padding = torch.zeros(
            (len(ids), side**2 - ids.shape[1]), dtype=torch.float64, device=ids.device
        )
        return torch.cat((config.omega * theta, padding), 1).reshape(len(ids), side, side)
    if config.method == "exponential":
        return (
            decay
            * torch.sin(config.omega * theta)[:, :, None]
            * torch.cos(config.omega * theta)[:, None]
            + theta[:, :, None]
        )
    if config.method == "diagonal":
        matrix = decay * torch.sin(config.omega * theta)[:, :, None] + theta[:, :, None]
        diagonal = torch.arange(ids.shape[1], device=ids.device)
        matrix[:, diagonal, diagonal] = torch.log1p(ids.double())
        return matrix
    modulation = (
        config.delta * theta
        if config.method == "phase"
        else (ids * config.prime) % config.hash_size
    )
    return (
        decay * torch.sin((config.omega * positions + modulation)[:, :, None]) + theta[:, :, None]
    )


def qcse_states(contexts: Tensor, size: int, config: ContextConfig | None = None) -> Tensor:
    config = config or ContextConfig()
    count = qubits(size)
    if contexts.ndim != 2 or torch.any(contexts < -1) or torch.any(contexts >= size):
        raise ValueError("Expected word IDs and -1 padding")
    lengths = (contexts >= 0).sum(1)
    if torch.any(lengths == 0):
        raise ValueError("QCSE needs at least one preceding token")
    result = torch.zeros((len(contexts), size), dtype=torch.complex128, device=contexts.device)
    for length in range(1, contexts.shape[1] + 1):
        rows = torch.where(lengths == length)[0]
        if not rows.numel():
            continue
        ids = contexts[rows, -length:]
        flat = context_matrices(ids, size, config).flatten(1)
        padding = torch.zeros(
            (len(rows), (-flat.shape[1]) % (2 * count)), dtype=torch.float64, device=contexts.device
        )
        angles = torch.cat((flat, padding), 1).reshape(len(rows), -1, count, 2)
        state = torch.full(
            (len(rows), size), size**-0.5, dtype=torch.complex128, device=contexts.device
        )
        for layer in range(angles.shape[1]):
            for q in range(count):
                state = rotate(state, q, angles[:, layer, q, 0], "x")
                state = rotate(state, q, angles[:, layer, q, 1], "z")
            for q in range(count - 1):
                state = cnot(state, q, q + 1)
        result = result.index_copy(0, rows, state)
    return result


def segmented_amplitudes(contexts: Tensor, size: int, recent: int = 4) -> Tensor:
    """Separate normalized recent/older streams in an extra sector qubit.

    Use fixed .7/.97 population decays. If older context is absent, all
    probability stays in the recent sector; padding never fabricates a token.
    """
    if not 1 <= recent <= contexts.shape[1]:
        raise ValueError("Recent window must fit the context")
    near = amplitudes(contexts[:, -recent:], size, decay=0.7)
    older = contexts[:, :-recent]
    rows = torch.where((older >= 0).any(dim=1))[0]
    far = torch.zeros_like(near)
    if rows.numel():
        far = far.index_copy(0, rows, amplitudes(older[rows], size, decay=0.97))
    combined = torch.cat((near, far), dim=1)
    return combined / combined.abs().square().sum(1, keepdim=True).sqrt()


def pooled_angles(contexts: Tensor, table: Tensor, recent: int = 4) -> Tensor:
    """Pool learned local angles for last word, other recent words and older words.

    Table axes are interleaving stage, stream, token and qubit. Fixed .7/.97
    population decays retain order within the two longer streams. Empty streams
    contribute zero, including padding; there is no learned classical word head.
    """
    if (
        contexts.ndim != 2
        or table.ndim != 4
        or table.shape[1] != 3
        or not 1 <= recent <= contexts.shape[1]
        or torch.any(contexts < -1)
        or torch.any(contexts >= table.shape[2])
    ):
        raise ValueError("Expected valid contexts and a stage/three-stream/token/qubit table")
    streams = (contexts[:, -1:], contexts[:, -recent:-1], contexts[:, :-recent])
    stages: list[Tensor] = []
    for stage in range(table.shape[0]):
        angles = table.new_zeros(len(contexts), table.shape[-1])
        for stream, (ids, decay) in enumerate(zip(streams, (1.0, 0.7, 0.97), strict=True)):
            ages = torch.arange(ids.shape[1] - 1, -1, -1, dtype=table.dtype, device=contexts.device)
            weights = (decay**ages) * (ids >= 0)
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
            angles = angles + (table[stage, stream, ids.clamp_min(0)] * weights[:, :, None]).sum(1)
        stages.append(angles)
    return torch.stack(stages, dim=1)


def product_states(angles: Tensor) -> Tensor:
    """Prepare independent RY rotations of zero qubits, in little-endian order."""
    if angles.ndim != 2 or angles.shape[1] < 1:
        raise ValueError("Expected one RY angle per example and qubit")
    states = angles.new_ones(len(angles), 1)
    for qubit in range(angles.shape[1]):
        angle = angles[:, qubit, None] / 2
        states = torch.cat((states * angle.cos(), states * angle.sin()), dim=1)
    return states.to(torch.complex128)
