"""Small quantum predictors and independently trainable classical controls."""

from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn

from .data import Dataset, bigram_probabilities
from .encoding import ContextConfig, ContextMethod, amplitudes, qcse_states, segmented_amplitudes
from .quantum import (
    ansatz,
    evolve,
    hermitian,
    independent_bits,
    qubits,
    readout,
    seed_random,
    spectral_action,
)

ModelKind = Literal[
    "segmented",
    "segmented_mixture",
    "segmented_mlp",
    "ancilla",
    "ancilla_dephased",
    "lowrank",
    "lowrank_static",
    "interleaved",
    "interleaved_static",
    "hamiltonian",
    "dephased",
    "qcse",
    "mlp",
    "hybrid",
    "classical_hybrid",
    "bigram",
    "calibrated_bigram",
    "unigram",
]
MODEL_KINDS: tuple[ModelKind, ...] = (
    "segmented",
    "segmented_mixture",
    "segmented_mlp",
    "ancilla",
    "ancilla_dephased",
    "lowrank",
    "lowrank_static",
    "interleaved",
    "interleaved_static",
    "hamiltonian",
    "dephased",
    "qcse",
    "mlp",
    "hybrid",
    "classical_hybrid",
    "bigram",
    "calibrated_bigram",
    "unigram",
)


@dataclass(frozen=True)
class ModelConfig:
    kind: ModelKind
    size: int
    window: int
    layers: int = 2
    encoder: Literal["auto", "amplitude", "qcse"] = "auto"
    context_method: ContextMethod = "exponential"
    decoder: Literal["born", "rotated", "marginal"] = "born"
    latent: int = 16
    embedding: int = 16
    freeze_core: bool = False
    ancilla_size: int = 2
    rank: int = 32
    phase_scale: float = 1.0
    amplitude_decay: float = 0.7
    recent_window: int = 4

    def __post_init__(self) -> None:
        if self.kind.startswith("segmented") and (
            self.encoder not in {"auto", "amplitude"}
            or self.decoder != "born"
            or self.ancilla_size != 2
            or not 1 <= self.recent_window <= self.window
        ):
            raise ValueError("Segmented models need two sectors and a valid recent window")
        if not 0 < self.amplitude_decay < 1:
            raise ValueError("Amplitude decay must lie strictly between zero and one")
        if self.rank < 1 or self.phase_scale < 0:
            raise ValueError("Rank must be positive and phase scale nonnegative")
        if self.ancilla_size != 1:
            qubits(self.ancilla_size)
        qubits(self.size)
        qubits(self.latent)
        if self.kind not in MODEL_KINDS or min(self.window, self.layers, self.embedding) < 1:
            raise ValueError("Invalid model configuration")


class WordModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def states(self, contexts: Tensor) -> Tensor:
        encoding = self.config.encoder
        if encoding == "auto":
            encoding = "qcse" if self.config.kind == "qcse" else "amplitude"
        if encoding == "qcse":
            return qcse_states(
                contexts, self.config.size, ContextConfig(self.config.context_method)
            )
        return amplitudes(contexts, self.config.size, decay=self.config.amplitude_decay)

    def forward(self, contexts: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    def protocol(self) -> str:
        if self.config.kind in {
            "hamiltonian",
            "dephased",
            "qcse",
            "lowrank",
            "lowrank_static",
            "interleaved",
            "interleaved_static",
        }:
            return (
                "independent-bit sampling (multiple preparations)"
                if self.config.decoder == "marginal"
                else "joint Born measurement"
            )
        if self.config.kind in {"ancilla", "ancilla_dephased", "segmented", "segmented_mixture"}:
            return "one word-register measurement; auxiliary register ignored"
        if self.config.kind == "hybrid":
            return "one latent measurement then classical conditional word sampling"
        return "classical categorical sampling; argmax is also available"


class HamiltonianModel(WordModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.coefficients = nn.Parameter(
            torch.randn(config.size**2, dtype=torch.float64) * 0.03,
            requires_grad=not config.freeze_core,
        )
        self.readout_angles = nn.Parameter(
            torch.zeros(qubits(config.size), 2, dtype=torch.float64),
            requires_grad=config.decoder == "rotated",
        )

    def operator(self) -> Tensor:
        operator = evolve(hermitian(self.coefficients, self.config.size))
        if self.config.decoder == "rotated":
            operator = readout(operator.T, self.readout_angles).T
        return operator

    def forward(self, contexts: Tensor) -> Tensor:
        states, operator = self.states(contexts), self.operator()
        probabilities = (
            states.abs().square() @ operator.abs().square().T
            if self.config.kind == "dephased"
            else (states @ operator.T).abs().square()
        )
        return (
            independent_bits(probabilities) if self.config.decoder == "marginal" else probabilities
        )


class InterleavedModel(WordModel):
    """Dense learned evolutions interleaved with fixed, local context RZ gates.

    Random token/position features are fixed independently of corpus statistics.
    Every valid position has equal scale; padding contributes zero. The static
    control retains the same trained parameters and initial state but omits RZ.
    """

    token_angles: Tensor
    position_signs: Tensor
    basis_z: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.encoder not in {"auto", "amplitude"} or config.decoder != "born":
            raise ValueError("Interleaving uses amplitude preparation and direct Born readout")
        count = qubits(config.size)
        fixed = torch.Generator().manual_seed(20261028)
        self.register_buffer(
            "token_angles",
            torch.randn(config.size, count, generator=fixed, dtype=torch.float64),
        )
        self.register_buffer(
            "position_signs",
            2
            * torch.randint(2, (config.layers - 1, config.window, count), generator=fixed).double()
            - 1,
        )
        indices = torch.arange(config.size)
        self.register_buffer(
            "basis_z", 1 - 2 * ((indices[:, None] >> torch.arange(count)) & 1).double()
        )
        self.initialize_evolution(config)

    def initialize_evolution(self, config: ModelConfig) -> None:
        self.coefficients = nn.Parameter(
            torch.randn(config.layers, config.size**2, dtype=torch.float64) * 0.03,
            requires_grad=not config.freeze_core,
        )

    def evolve_states(self, states: Tensor, layer: int) -> Tensor:
        return states @ evolve(hermitian(self.coefficients[layer], self.config.size)).T

    def forward(self, contexts: Tensor) -> Tensor:
        states = self.states(contexts)
        valid = contexts >= 0
        features = self.token_angles[contexts.clamp_min(0)] * valid[:, :, None]
        scale = valid.sum(1).clamp_min(1).sqrt()[:, None]
        for layer in range(self.config.layers):
            states = self.evolve_states(states, layer)
            if layer < self.config.layers - 1 and self.config.kind in {"interleaved", "lowrank"}:
                angles = (features * self.position_signs[layer]).sum(1) / scale
                states = states * torch.exp(
                    -0.5j * (self.config.phase_scale * angles @ self.basis_z.T)
                )
        return states.abs().square()


class LowRankModel(InterleavedModel):
    """Learn an orthonormal active subspace and its Hamiltonian eigenvalues."""

    def initialize_evolution(self, config: ModelConfig) -> None:
        rank = min(config.rank, config.size)
        self.factors = nn.Parameter(
            torch.randn(config.layers, config.size, rank, 2, dtype=torch.float64)
            / config.size**0.5,
            requires_grad=not config.freeze_core,
        )
        self.energies = nn.Parameter(
            torch.empty(config.layers, rank, dtype=torch.float64).uniform_(-torch.pi, torch.pi),
            requires_grad=not config.freeze_core,
        )

    def evolve_states(self, states: Tensor, layer: int) -> Tensor:
        return spectral_action(states, self.factors[layer], self.energies[layer])


class AncillaModel(WordModel):
    """Low-rank joint evolution with an initialized, unobserved auxiliary register."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.encoder not in {"auto", "amplitude"} or config.decoder != "born":
            raise ValueError("Ancilla models require amplitude input and word Born measurement")
        self.dimension = config.size * config.ancilla_size
        rank = min(config.rank, self.dimension)
        self.factors = nn.Parameter(
            torch.randn(config.layers, self.dimension, rank, 2, dtype=torch.float64)
            / self.dimension**0.5,
            requires_grad=not config.freeze_core,
        )
        self.energies = nn.Parameter(
            torch.empty(config.layers, rank, dtype=torch.float64).uniform_(-torch.pi, torch.pi),
            requires_grad=not config.freeze_core,
        )

    def evolved(self, inputs: Tensor) -> Tensor:
        # Word index is the low-order register; auxiliary starts in |0>.
        states = torch.cat(
            (inputs, inputs.new_zeros(len(inputs), self.dimension - inputs.shape[1])), dim=1
        )
        for layer in range(self.config.layers):
            states = spectral_action(states, self.factors[layer], self.energies[layer])
        return states

    def word_probabilities(self, states: Tensor) -> Tensor:
        return (
            states.abs()
            .square()
            .reshape(len(states), self.config.ancilla_size, self.config.size)
            .sum(1)
        )

    def forward(self, contexts: Tensor) -> Tensor:
        states = self.states(contexts)
        if self.config.kind == "ancilla_dephased":
            basis = torch.eye(self.config.size, dtype=torch.complex128, device=states.device)
            transitions = self.word_probabilities(self.evolved(basis))
            return states.abs().square() @ transitions
        return self.word_probabilities(self.evolved(states))


class SegmentedModel(AncillaModel):
    """Sector-coherent input or independently retrained sector-dephased control."""

    def states(self, contexts: Tensor) -> Tensor:
        return segmented_amplitudes(contexts, self.config.size, self.config.recent_window)

    def forward(self, contexts: Tensor) -> Tensor:
        states = self.states(contexts)
        if self.config.kind == "segmented_mixture":
            zero = torch.zeros_like(states[:, : self.config.size])
            near = torch.cat((states[:, : self.config.size], zero), 1)
            far = torch.cat((zero, states[:, self.config.size :]), 1)
            evolved = self.word_probabilities(self.evolved(torch.cat((near, far), 0)))
            return evolved[: len(states)] + evolved[len(states) :]
        return self.word_probabilities(self.evolved(states))


class SegmentedMLP(WordModel):
    """Parameter-matched classical reference seeing the same full complex input."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        budget = config.layers * (4 * config.size + 1) * min(config.rank, 2 * config.size)
        hidden = max(1, (budget - config.size) // (5 * config.size + 1))
        self.network = nn.Sequential(
            nn.Linear(4 * config.size, hidden), nn.Tanh(), nn.Linear(hidden, config.size)
        ).double()

    def states(self, contexts: Tensor) -> Tensor:
        return segmented_amplitudes(contexts, self.config.size, self.config.recent_window)

    def forward(self, contexts: Tensor) -> Tensor:
        states = self.states(contexts)
        return cast(Tensor, self.network(torch.cat((states.real, states.imag), 1))).softmax(-1)


class QCSEModel(WordModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.weights = nn.Parameter(
            torch.empty(config.layers, 3 * qubits(config.size) - 1, dtype=torch.float64).uniform_(
                -torch.pi, torch.pi
            ),
            requires_grad=not config.freeze_core,
        )
        self.readout_angles = nn.Parameter(
            torch.zeros(qubits(config.size), 2, dtype=torch.float64),
            requires_grad=config.decoder == "rotated",
        )

    def forward(self, contexts: Tensor) -> Tensor:
        states = ansatz(self.states(contexts), self.weights)
        if self.config.decoder == "rotated":
            states = readout(states, self.readout_angles)
        probabilities = states.abs().square()
        return (
            independent_bits(probabilities) if self.config.decoder == "marginal" else probabilities
        )


class ClassicalMLP(WordModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        hidden = max(1, (config.size**2 - config.size) // (3 * config.size + 1))
        self.network = nn.Sequential(
            nn.Linear(2 * config.size, hidden), nn.Tanh(), nn.Linear(hidden, config.size)
        ).double()

    def forward(self, contexts: Tensor) -> Tensor:
        states = self.states(contexts)
        return cast(Tensor, self.network(torch.cat((states.real, states.imag), 1))).softmax(-1)


class ContextEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.padding = config.size
        self.lookup = nn.Embedding(
            config.size + 1, config.embedding, padding_idx=config.size
        ).double()
        self.network = nn.Sequential(
            nn.Linear(config.window * config.embedding, 32),
            nn.Tanh(),
            nn.Linear(32, config.latent - 1),
        ).double()

    def forward(self, contexts: Tensor) -> Tensor:
        ids = torch.where(contexts >= 0, contexts, self.padding)
        features = cast(Tensor, self.lookup(ids)).flatten(1)
        raw = cast(Tensor, self.network(features))
        # A constant coordinate prevents a zero-norm prepared state.
        state = torch.cat((torch.ones(len(raw), 1, dtype=torch.float64), raw), 1)
        norm = state.square().sum(dim=1, keepdim=True).sqrt()
        return (state / norm).to(torch.complex128)


class ClassicalDecoder(nn.Module):
    """P(word | measured latent bitstring); no access to amplitudes or raw context."""

    def __init__(self, latent: int, vocabulary: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.randn(latent, vocabulary, dtype=torch.float64) * 0.01)

    def forward(self, probabilities: Tensor) -> Tensor:
        return probabilities @ self.logits.softmax(-1)


class HybridModel(WordModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.encoder = ContextEncoder(config)
        self.coefficients = nn.Parameter(
            torch.randn(config.latent**2, dtype=torch.float64) * 0.03,
            requires_grad=not config.freeze_core,
        )
        self.decoder = ClassicalDecoder(config.latent, config.size)

    def forward(self, contexts: Tensor) -> Tensor:
        states = cast(Tensor, self.encoder(contexts))
        if self.config.kind == "classical_hybrid":
            probabilities = states.abs().square() @ self.coefficients.reshape(
                self.config.latent, -1
            ).softmax(-1)
        else:
            operator = evolve(hermitian(self.coefficients, self.config.latent))
            probabilities = (states @ operator.T).abs().square()
        return cast(Tensor, self.decoder(probabilities))


class CountModel(WordModel):
    prior: Tensor
    unigram: Tensor

    def __init__(self, config: ModelConfig, dataset: Dataset) -> None:
        super().__init__(config)
        self.register_buffer(
            "prior", torch.tensor(bigram_probabilities(dataset.transitions, dataset.unigram))
        )
        self.register_buffer("unigram", torch.tensor(dataset.unigram))
        learn = config.kind == "calibrated_bigram"
        self.log_temperature = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float64), requires_grad=learn
        )
        self.mixture_logit = nn.Parameter(
            torch.tensor(-6.0, dtype=torch.float64), requires_grad=learn
        )

    def forward(self, contexts: Tensor) -> Tensor:
        if self.config.kind == "unigram":
            return self.unigram.expand(len(contexts), -1)
        probabilities = self.prior[contexts[:, -1]]
        if self.config.kind == "calibrated_bigram":
            probabilities = (probabilities.log() / self.log_temperature.exp()).softmax(-1)
            mixture = self.mixture_logit.sigmoid()
            probabilities = (1 - mixture) * probabilities + mixture * self.unigram
        return probabilities


def build_model(config: ModelConfig, dataset: Dataset, seed: int) -> WordModel:
    if config.size != dataset.size or config.window != dataset.window:
        raise ValueError("Model and dataset dimensions differ")
    seed_random(seed)
    if config.kind in {"segmented", "segmented_mixture"}:
        return SegmentedModel(config)
    if config.kind == "segmented_mlp":
        return SegmentedMLP(config)
    if config.kind in {"ancilla", "ancilla_dephased"}:
        return AncillaModel(config)
    if config.kind in {"lowrank", "lowrank_static"}:
        return LowRankModel(config)
    if config.kind in {"interleaved", "interleaved_static"}:
        return InterleavedModel(config)
    if config.kind in {"hamiltonian", "dephased"}:
        return HamiltonianModel(config)
    if config.kind == "qcse":
        return QCSEModel(config)
    if config.kind == "mlp":
        return ClassicalMLP(config)
    if config.kind in {"hybrid", "classical_hybrid"}:
        return HybridModel(config)
    return CountModel(config, dataset)
