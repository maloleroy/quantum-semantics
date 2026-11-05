"""Small quantum predictors and independently trainable classical controls."""

from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn

from .data import Dataset, bigram_probabilities
from .device import execution_device
from .encoding import (
    ContextConfig,
    ContextMethod,
    amplitudes,
    bloch_coordinates,
    factor_product,
    pooled_angles,
    product_states,
    qcse_states,
    segmented_amplitudes,
    sequential_factors,
)
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
    "prior_position",
    "prior_position_mlp",
    "prior",
    "prior_unigram",
    "prior_mlp",
    "adaptive",
    "adaptive_mlp",
    "stream",
    "stream_pooled",
    "stream_mlp",
    "sequence",
    "sequence_pooled",
    "sequence_fixed",
    "sequence_mlp",
    "product",
    "product_fixed",
    "product_mlp",
    "conditional",
    "conditional_fixed",
    "conditional_mlp",
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
    "prior_position",
    "prior_position_mlp",
    "prior",
    "prior_unigram",
    "prior_mlp",
    "adaptive",
    "adaptive_mlp",
    "stream",
    "stream_pooled",
    "stream_mlp",
    "sequence",
    "sequence_pooled",
    "sequence_fixed",
    "sequence_mlp",
    "product",
    "product_fixed",
    "product_mlp",
    "conditional",
    "conditional_fixed",
    "conditional_mlp",
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
        if self.kind.startswith("prior") and (
            self.encoder != "auto"
            or self.decoder != "born"
            or self.layers < 2
            or not 1 <= self.recent_window <= self.window
        ):
            raise ValueError("Prior models require auto encoding, Born readout and context stages")
        if self.kind in {"prior_mlp", "prior_position_mlp"} and self.freeze_core:
            raise ValueError("Freezing Hamiltonian bases is only defined for quantum prior models")
        if self.kind.startswith(("product", "sequence", "stream", "adaptive")) and (
            self.encoder != "auto"
            or self.decoder
            not in ({"born", "rotated"} if self.kind in {"product", "product_fixed"} else {"born"})
            or self.layers < 2
            or not 1 <= self.recent_window <= self.window
        ):
            raise ValueError("Local preparation models require auto encoding and quantum readout")
        if self.kind.startswith("stream") and self.window % self.layers:
            raise ValueError("Streaming context must divide evenly into Hamiltonian stages")
        if self.kind.startswith("conditional") and (
            self.encoder not in {"auto", "amplitude"}
            or self.decoder != "born"
            or self.layers < 2
            or not 1 <= self.recent_window <= self.window
        ):
            raise ValueError("Conditional models need amplitude input, Born readout and stages")
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

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

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
        if self.config.kind in {
            "prior_position",
            "prior",
            "prior_unigram",
            "adaptive",
            "stream",
            "stream_pooled",
            "sequence",
            "sequence_pooled",
            "sequence_fixed",
            "product",
            "product_fixed",
            "conditional",
            "conditional_fixed",
            "ancilla",
            "ancilla_dephased",
            "segmented",
            "segmented_mixture",
        }:
            return "one vocabulary-register measurement; auxiliary register ignored"
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
        if config.encoder not in {"auto", "amplitude"} or config.decoder not in (
            {"born", "rotated"} if config.kind in {"product", "product_fixed"} else {"born"}
        ):
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


def initial_control_angles(config: ModelConfig) -> Tensor:
    count = qubits(config.size * config.ancilla_size)
    return torch.randn(config.layers - 1, 3, config.size, count, dtype=torch.float64) * 0.1


class ConditionalModel(AncillaModel):
    """Train small token-angle tables between joint Hamiltonian evolutions."""

    basis_z: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.token_angles = nn.Parameter(
            initial_control_angles(config), requires_grad=config.kind != "conditional_fixed"
        )
        indices = torch.arange(self.dimension)
        self.register_buffer(
            "basis_z",
            1 - 2 * ((indices[:, None] >> torch.arange(qubits(self.dimension))) & 1).double(),
        )

    def forward(self, contexts: Tensor) -> Tensor:
        inputs = self.states(contexts)
        states = torch.cat(
            (inputs, inputs.new_zeros(len(inputs), self.dimension - inputs.shape[1])), 1
        )
        angles = self.config.phase_scale * pooled_angles(
            contexts, self.token_angles, self.config.recent_window
        )
        for layer in range(self.config.layers):
            states = spectral_action(states, self.factors[layer], self.energies[layer])
            if layer < self.config.layers - 1:
                states = states * torch.exp(-0.5j * (angles[:, layer] @ self.basis_z.T))
        return self.word_probabilities(states)


class ConditionalMLP(WordModel):
    """Classical control with the same amplitude and trainable pooled-angle inputs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.token_angles = nn.Parameter(initial_control_angles(config))
        dimension = config.size * config.ancilla_size
        budget = config.layers * (2 * dimension + 1) * min(config.rank, dimension)
        features = 2 * config.size + (config.layers - 1) * qubits(dimension)
        hidden = max(1, (budget - config.size) // (features + config.size + 1))
        self.network = nn.Sequential(
            nn.Linear(features, hidden), nn.Tanh(), nn.Linear(hidden, config.size)
        ).double()

    def forward(self, contexts: Tensor) -> Tensor:
        states = self.states(contexts)
        angles = self.config.phase_scale * pooled_angles(
            contexts, self.token_angles, self.config.recent_window
        )
        inputs = torch.cat((states.real, states.imag, angles.flatten(1)), 1)
        return cast(Tensor, self.network(inputs)).softmax(-1)


def initial_product_angles(config: ModelConfig) -> Tensor:
    count = qubits(config.size * config.ancilla_size)
    return torch.randn(config.layers, 3, config.size, count, dtype=torch.float64) * 0.1


def product_inputs(
    contexts: Tensor, table: Tensor, offsets: Tensor, config: ModelConfig
) -> tuple[Tensor, Tensor]:
    controls = offsets + config.phase_scale * pooled_angles(contexts, table, config.recent_window)
    return product_states(controls[:, 0]), controls[:, 1:]


class ProductModel(AncillaModel):
    """Local learned preparation, joint Hamiltonians and direct vocabulary measurement."""

    basis_z: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        train_controls = config.kind != "product_fixed"
        self.token_angles = nn.Parameter(
            initial_product_angles(config), requires_grad=train_controls
        )
        self.control_offsets = nn.Parameter(
            torch.zeros(config.layers, qubits(self.dimension), dtype=torch.float64),
            requires_grad=train_controls,
        )
        self.readout_angles = (
            nn.Parameter(torch.zeros(qubits(config.size), 2, dtype=torch.float64))
            if config.decoder == "rotated"
            else None
        )
        indices = torch.arange(self.dimension)
        self.register_buffer(
            "basis_z",
            1 - 2 * ((indices[:, None] >> torch.arange(qubits(self.dimension))) & 1).double(),
        )

    def states(self, contexts: Tensor) -> Tensor:
        return product_inputs(contexts, self.token_angles, self.control_offsets, self.config)[0]

    def forward(self, contexts: Tensor) -> Tensor:
        states, controls = product_inputs(
            contexts, self.token_angles, self.control_offsets, self.config
        )
        for layer in range(self.config.layers):
            states = spectral_action(states, self.factors[layer], self.energies[layer])
            if layer < self.config.layers - 1:
                states = states * torch.exp(-0.5j * (controls[:, layer] @ self.basis_z.T))
        return self.word_probabilities(states)

    def word_probabilities(self, states: Tensor) -> Tensor:
        if self.readout_angles is not None:
            # Identical word-register rotation within each auxiliary sector.
            rotated = readout(states.reshape(-1, self.config.size), self.readout_angles)
            states = rotated.reshape_as(states)
        return super().word_probabilities(states)


def initial_spectral_angles(config: ModelConfig) -> Tensor:
    rank = min(config.rank, config.size * config.ancilla_size)
    return torch.zeros(1, 3, config.size, rank, dtype=torch.float64)


def spectral_controls(contexts: Tensor, table: Tensor, config: ModelConfig) -> Tensor:
    """Shared linear context pooling gives eigenvalue shifts in each learned spectral basis."""
    return config.phase_scale * pooled_angles(contexts, table, config.recent_window)[:, 0]


class AdaptiveModel(ProductModel):
    """Direct measurement after Hamiltonians with context-dependent eigenvalues."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.spectral_angles = nn.Parameter(initial_spectral_angles(config))

    def forward(self, contexts: Tensor) -> Tensor:
        states, controls = product_inputs(
            contexts, self.token_angles, self.control_offsets, self.config
        )
        shifts = spectral_controls(contexts, self.spectral_angles, self.config)
        for layer in range(self.config.layers):
            states = spectral_action(states, self.factors[layer], self.energies[layer] + shifts)
            if layer < self.config.layers - 1:
                states = states * torch.exp(-0.5j * (controls[:, layer] @ self.basis_z.T))
        return self.word_probabilities(states)


class AdaptiveMLP(WordModel):
    """Matched classical map receiving preparation, local controls and spectral shifts."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        dimension = config.size * config.ancilla_size
        count = qubits(dimension)
        rank = min(config.rank, dimension)
        self.token_angles = nn.Parameter(initial_product_angles(config))
        self.control_offsets = nn.Parameter(torch.zeros(config.layers, count, dtype=torch.float64))
        self.spectral_angles = nn.Parameter(initial_spectral_angles(config))
        budget = config.layers * (2 * dimension + 1) * rank
        features = dimension + (config.layers - 1) * count + rank
        hidden = max(1, (budget - config.size) // (features + config.size + 1))
        self.network = nn.Sequential(
            nn.Linear(features, hidden), nn.Tanh(), nn.Linear(hidden, config.size)
        ).double()

    def states(self, contexts: Tensor) -> Tensor:
        return product_inputs(contexts, self.token_angles, self.control_offsets, self.config)[0]

    def forward(self, contexts: Tensor) -> Tensor:
        states, controls = product_inputs(
            contexts, self.token_angles, self.control_offsets, self.config
        )
        shifts = spectral_controls(contexts, self.spectral_angles, self.config)
        inputs = torch.cat((states.real, controls.flatten(1), shifts), 1)
        return cast(Tensor, self.network(inputs)).softmax(-1)


def prior_probabilities(dataset: Dataset, unigram_only: bool = False) -> Tensor:
    """Fixed training-count rows; the final row handles a padded last position."""
    counts = torch.tensor(bigram_probabilities(dataset.transitions, dataset.unigram))
    unigram = torch.tensor(dataset.unigram)[None]
    if unigram_only:
        counts = unigram.expand_as(counts)
    return torch.cat((counts, unigram), 0)


def prior_position_logits(config: ModelConfig, table: Tensor) -> nn.Parameter | None:
    """Feature-specific relative-position weights, with fixed-pooling initialization."""
    if config.kind not in {"prior_position", "prior_position_mlp"}:
        return None
    return nn.Parameter(table.new_zeros(table.shape[0], config.window - 1, table.shape[-1]))


class PriorModel(AncillaModel):
    """Refine fixed count amplitudes with contextual Hamiltonians and Born measurement."""

    prior: Tensor
    basis_z: Tensor

    def __init__(self, config: ModelConfig, dataset: Dataset) -> None:
        super().__init__(config)
        self.register_buffer("prior", prior_probabilities(dataset, config.kind == "prior_unigram"))
        self.token_angles = nn.Parameter(initial_control_angles(config))
        self.spectral_angles = nn.Parameter(initial_spectral_angles(config))
        self.position_angles = prior_position_logits(config, self.token_angles)
        self.position_spectra = prior_position_logits(config, self.spectral_angles)
        indices = torch.arange(self.dimension)
        self.register_buffer(
            "basis_z",
            1 - 2 * ((indices[:, None] >> torch.arange(qubits(self.dimension))) & 1).double(),
        )
        # Identity evolutions initially reproduce the count distribution exactly.
        # Interleaved RZ gates only change phases until an evolution learns to mix them.
        with torch.no_grad():
            self.energies.zero_()

    def states(self, contexts: Tensor) -> Tensor:
        return self.prior[contexts[:, -1]].sqrt().to(torch.complex128)

    def forward(self, contexts: Tensor) -> Tensor:
        inputs = self.states(contexts)
        states = torch.cat(
            (inputs, inputs.new_zeros(len(inputs), self.dimension - self.config.size)), 1
        )
        angles = self.config.phase_scale * pooled_angles(
            contexts, self.token_angles, self.config.recent_window, self.position_angles
        )
        shifts = (
            self.config.phase_scale
            * pooled_angles(
                contexts, self.spectral_angles, self.config.recent_window, self.position_spectra
            )[:, 0]
        )
        for layer in range(self.config.layers):
            states = spectral_action(states, self.factors[layer], self.energies[layer] + shifts)
            if layer < self.config.layers - 1:
                states = states * torch.exp(-0.5j * (angles[:, layer] @ self.basis_z.T))
        return self.word_probabilities(states)


class PriorMLP(WordModel):
    """Same fixed prior and context features, with a parameter-matched logit correction."""

    prior: Tensor

    def __init__(self, config: ModelConfig, dataset: Dataset) -> None:
        super().__init__(config)
        self.register_buffer("prior", prior_probabilities(dataset))
        self.token_angles = nn.Parameter(initial_control_angles(config))
        self.spectral_angles = nn.Parameter(initial_spectral_angles(config))
        self.position_angles = prior_position_logits(config, self.token_angles)
        self.position_spectra = prior_position_logits(config, self.spectral_angles)
        dimension = config.size * config.ancilla_size
        rank = min(config.rank, dimension)
        budget = config.layers * (2 * dimension + 1) * rank
        features = config.size + (config.layers - 1) * qubits(dimension) + rank
        hidden = max(1, (budget - config.size) // (features + config.size + 1))
        self.hidden = nn.Linear(features, hidden).double()
        self.correction = nn.Linear(hidden, config.size).double()
        with torch.no_grad():
            self.correction.weight.zero_()
            self.correction.bias.zero_()

    def forward(self, contexts: Tensor) -> Tensor:
        prior = self.prior[contexts[:, -1]]
        angles = self.config.phase_scale * pooled_angles(
            contexts, self.token_angles, self.config.recent_window, self.position_angles
        )
        shifts = (
            self.config.phase_scale
            * pooled_angles(
                contexts, self.spectral_angles, self.config.recent_window, self.position_spectra
            )[:, 0]
        )
        inputs = torch.cat((prior.sqrt(), angles.flatten(1), shifts), 1)
        logits = self.correction(self.hidden(inputs).tanh())
        return cast(Tensor, (prior.log() + logits).softmax(-1))


class ProductMLP(WordModel):
    """Classical control receiving the same prepared real amplitudes and later RZ angles."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        dimension = config.size * config.ancilla_size
        count = qubits(dimension)
        self.token_angles = nn.Parameter(initial_product_angles(config))
        self.control_offsets = nn.Parameter(torch.zeros(config.layers, count, dtype=torch.float64))
        budget = config.layers * (2 * dimension + 1) * min(config.rank, dimension)
        features = dimension + (config.layers - 1) * count
        hidden = max(1, (budget - config.size) // (features + config.size + 1))
        self.network = nn.Sequential(
            nn.Linear(features, hidden), nn.Tanh(), nn.Linear(hidden, config.size)
        ).double()

    def states(self, contexts: Tensor) -> Tensor:
        return product_inputs(contexts, self.token_angles, self.control_offsets, self.config)[0]

    def forward(self, contexts: Tensor) -> Tensor:
        states, controls = product_inputs(
            contexts, self.token_angles, self.control_offsets, self.config
        )
        inputs = torch.cat((states.real, controls.flatten(1)), 1)
        return cast(Tensor, self.network(inputs)).softmax(-1)


def sequence_inputs(
    contexts: Tensor, sequence_table: Tensor, table: Tensor, offsets: Tensor, config: ModelConfig
) -> tuple[Tensor, Tensor, Tensor]:
    zero, one = sequential_factors(
        contexts,
        sequence_table,
        offsets[:2],
        config.phase_scale,
        pooled=config.kind == "sequence_pooled",
    )
    controls = offsets[2:] + config.phase_scale * pooled_angles(
        contexts, table, config.recent_window
    )
    return zero, one, controls


def initial_sequence_angles(config: ModelConfig) -> Tensor:
    return (
        torch.randn(2, config.size, qubits(config.size * config.ancilla_size), dtype=torch.float64)
        * 0.1
    )


class SequenceModel(AncillaModel):
    """Token-ordered local preparation followed by joint Hamiltonian dynamics."""

    basis_z: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        train_controls = config.kind != "sequence_fixed"
        count = qubits(self.dimension)
        self.sequence_angles = nn.Parameter(
            initial_sequence_angles(config), requires_grad=train_controls
        )
        self.token_angles = nn.Parameter(
            initial_control_angles(config), requires_grad=train_controls
        )
        self.control_offsets = nn.Parameter(
            torch.zeros(config.layers + 1, count, dtype=torch.float64), requires_grad=train_controls
        )
        indices = torch.arange(self.dimension)
        self.register_buffer(
            "basis_z", 1 - 2 * ((indices[:, None] >> torch.arange(count)) & 1).double()
        )

    def states(self, contexts: Tensor) -> Tensor:
        zero, one, _ = sequence_inputs(
            contexts, self.sequence_angles, self.token_angles, self.control_offsets, self.config
        )
        return factor_product(zero, one)

    def forward(self, contexts: Tensor) -> Tensor:
        zero, one, controls = sequence_inputs(
            contexts, self.sequence_angles, self.token_angles, self.control_offsets, self.config
        )
        states = factor_product(zero, one)
        for layer in range(self.config.layers):
            states = spectral_action(states, self.factors[layer], self.energies[layer])
            if layer < self.config.layers - 1:
                states = states * torch.exp(-0.5j * (controls[:, layer] @ self.basis_z.T))
        return self.word_probabilities(states)


class SequenceMLP(WordModel):
    """Classical control using the same pure product state, without global-phase information.

    Three Bloch components per local factor fully describe the prepared quantum
    state. Constructing this compact description is classical feature processing,
    not a claim of extracting all components from one physical preparation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        dimension = config.size * config.ancilla_size
        count = qubits(dimension)
        self.sequence_angles = nn.Parameter(initial_sequence_angles(config))
        self.token_angles = nn.Parameter(initial_control_angles(config))
        self.control_offsets = nn.Parameter(
            torch.zeros(config.layers + 1, count, dtype=torch.float64)
        )
        budget = config.layers * (2 * dimension + 1) * min(config.rank, dimension)
        features = (config.layers + 2) * count
        hidden = max(1, (budget - config.size) // (features + config.size + 1))
        self.network = nn.Sequential(
            nn.Linear(features, hidden), nn.Tanh(), nn.Linear(hidden, config.size)
        ).double()

    def states(self, contexts: Tensor) -> Tensor:
        zero, one, _ = sequence_inputs(
            contexts, self.sequence_angles, self.token_angles, self.control_offsets, self.config
        )
        return factor_product(zero, one)

    def forward(self, contexts: Tensor) -> Tensor:
        zero, one, controls = sequence_inputs(
            contexts, self.sequence_angles, self.token_angles, self.control_offsets, self.config
        )
        bloch = bloch_coordinates(zero, one)
        inputs = torch.cat((bloch.flatten(1), controls.flatten(1)), 1)
        return cast(Tensor, self.network(inputs)).softmax(-1)


def stream_inputs(
    contexts: Tensor, table: Tensor, offsets: Tensor, config: ModelConfig
) -> tuple[Tensor, Tensor]:
    """Use the oldest block for RY preparation; later blocks supply shared-table RZ gates."""
    blocks = (
        (contexts,) * config.layers
        if config.kind == "stream_pooled"
        else tuple(
            contexts[:, start : start + config.window // config.layers]
            for start in range(0, config.window, config.window // config.layers)
        )
    )
    controls: list[Tensor] = []
    for stage, block in enumerate(blocks):
        slot = 0 if stage == 0 else 1
        angles = pooled_angles(
            block, table[slot : slot + 1], min(config.recent_window, block.shape[1])
        )[:, 0]
        controls.append(offsets[stage] + config.phase_scale * angles)
    return product_states(controls[0]), torch.stack(controls[1:], dim=1)


def initial_stream_controls(config: ModelConfig) -> tuple[Tensor, Tensor]:
    count = qubits(config.size * config.ancilla_size)
    table = torch.randn(2, 3, config.size, count, dtype=torch.float64) * 0.1
    offsets = torch.zeros(config.layers, count, dtype=torch.float64)
    offsets[0] = torch.pi / 2
    return table, offsets


class StreamModel(AncillaModel):
    """Interleave successive context blocks with independently learned Hamiltonians."""

    basis_z: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        table, offsets = initial_stream_controls(config)
        self.token_angles = nn.Parameter(table)
        self.control_offsets = nn.Parameter(offsets)
        indices = torch.arange(self.dimension)
        self.register_buffer(
            "basis_z",
            1 - 2 * ((indices[:, None] >> torch.arange(qubits(self.dimension))) & 1).double(),
        )

    def states(self, contexts: Tensor) -> Tensor:
        return stream_inputs(contexts, self.token_angles, self.control_offsets, self.config)[0]

    def forward(self, contexts: Tensor) -> Tensor:
        states, controls = stream_inputs(
            contexts, self.token_angles, self.control_offsets, self.config
        )
        for stage in range(self.config.layers):
            if stage:
                states = states * torch.exp(-0.5j * (controls[:, stage - 1] @ self.basis_z.T))
            states = spectral_action(states, self.factors[stage], self.energies[stage])
        return self.word_probabilities(states)


class StreamMLP(WordModel):
    """Matched classical control seeing initial amplitudes and every later block's angles."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        table, offsets = initial_stream_controls(config)
        self.token_angles = nn.Parameter(table)
        self.control_offsets = nn.Parameter(offsets)
        dimension = config.size * config.ancilla_size
        features = dimension + (config.layers - 1) * qubits(dimension)
        budget = config.layers * (2 * dimension + 1) * min(config.rank, dimension)
        hidden = max(1, (budget - config.size) // (features + config.size + 1))
        self.network = nn.Sequential(
            nn.Linear(features, hidden), nn.Tanh(), nn.Linear(hidden, config.size)
        ).double()

    def states(self, contexts: Tensor) -> Tensor:
        return stream_inputs(contexts, self.token_angles, self.control_offsets, self.config)[0]

    def forward(self, contexts: Tensor) -> Tensor:
        states, controls = stream_inputs(
            contexts, self.token_angles, self.control_offsets, self.config
        )
        return cast(Tensor, self.network(torch.cat((states.real, controls.flatten(1)), 1))).softmax(
            -1
        )


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
        state = torch.cat((raw.new_ones(len(raw), 1), raw), 1)
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


def build_model(
    config: ModelConfig, dataset: Dataset, seed: int, device: str | torch.device = "cpu"
) -> WordModel:
    target = execution_device(device)
    return _initialize_model(config, dataset, seed).to(target)


def _initialize_model(config: ModelConfig, dataset: Dataset, seed: int) -> WordModel:
    if config.size != dataset.size or config.window != dataset.window:
        raise ValueError("Model and dataset dimensions differ")
    seed_random(seed)
    if config.kind in {"prior", "prior_unigram", "prior_position"}:
        return PriorModel(config, dataset)
    if config.kind in {"prior_mlp", "prior_position_mlp"}:
        return PriorMLP(config, dataset)
    if config.kind == "adaptive":
        return AdaptiveModel(config)
    if config.kind == "adaptive_mlp":
        return AdaptiveMLP(config)
    if config.kind in {"stream", "stream_pooled"}:
        return StreamModel(config)
    if config.kind == "stream_mlp":
        return StreamMLP(config)
    if config.kind in {"sequence", "sequence_pooled", "sequence_fixed"}:
        return SequenceModel(config)
    if config.kind == "sequence_mlp":
        return SequenceMLP(config)
    if config.kind in {"product", "product_fixed"}:
        return ProductModel(config)
    if config.kind == "product_mlp":
        return ProductMLP(config)
    if config.kind in {"conditional", "conditional_fixed"}:
        return ConditionalModel(config)
    if config.kind == "conditional_mlp":
        return ConditionalMLP(config)
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
