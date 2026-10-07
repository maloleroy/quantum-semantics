"""Task-matched adapters; existing research models/checkpoints stay independent."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .attention import QuantumAttentionModel, QuantumRegister
from .context import ContextConfig, context_matrix, encoding_angles
from .semantic import SemanticModel

ANSATZES = ("crz-chain", "crz-ring", "zz-chain", "xx-chain", "ry-cx", "local", "identity")


class VariationalCircuit(nn.Module):
    """Exact ordered gates; real/imaginary storage also supports Apple MPS."""

    def __init__(self, qubits, layers, kind="crz-chain", scale=0.05):
        super().__init__()
        if kind not in ANSATZES or min(qubits, layers) < 1:
            raise ValueError("Invalid circuit kind or dimensions")
        self.register = QuantumRegister(qubits)
        self.kind, self.qubits = kind, qubits
        self.edges = [(q, q + 1) for q in range(qubits - 1)]
        if kind == "crz-ring" and qubits > 2:
            self.edges.append((qubits - 1, 0))
        if kind in ("local", "identity", "ry-cx"):
            parameter_edges = []
        else:
            parameter_edges = self.edges
        width = 0 if kind == "identity" else 2 * qubits + len(parameter_edges)
        self.weights = nn.Parameter(scale * torch.randn(layers, width), requires_grad=width > 0)

    def forward(self, state):
        if self.kind == "identity":
            return state
        z = self.register.z
        for weights in self.weights:
            for q in range(self.qubits):
                state = self.register.rotate(
                    state, weights[q], q, "y" if self.kind == "ry-cx" else "x"
                )
            state = self.register.phase(state, weights[self.qubits : 2 * self.qubits])
            if self.kind == "ry-cx":
                state = self.register.entangle(state)
            elif self.kind not in ("local", "identity"):
                for index, (control, target) in enumerate(self.edges):
                    angle = weights[2 * self.qubits + index]
                    if self.kind == "xx-chain":
                        flipped = state[
                            :, self.register.flips[control][self.register.flips[target]]
                        ]
                        real, imag = state.unbind(-1)
                        fr, fi = flipped.unbind(-1)
                        c, s = torch.cos(angle / 2), torch.sin(angle / 2)
                        state = torch.stack((c * real + s * fi, c * imag - s * fr), -1)
                    else:
                        signs = (
                            (1 - z[:, control]) / 2 * z[:, target]
                            if self.kind.startswith("crz")
                            else z[:, control] * z[:, target]
                        )
                        state = SemanticModel.phase(state, -angle / 2 * signs)
        return state


class PaperQCSE(nn.Module):
    """Paper encoder/Z readout with autograd and explicit independent-bit decoding."""

    word_bits: torch.Tensor
    id_mapping: torch.Tensor

    def __init__(self, vocabulary_size, arm):
        super().__init__()
        qubits = (vocabulary_size - 1).bit_length()
        if not 1 <= qubits <= 16:
            raise ValueError("Paper adapter supports vocabulary sizes 2..65536")
        self.vocabulary_size = vocabulary_size
        self.context = ContextConfig(method=arm.get("context", "exponential"))
        self.circuit = VariationalCircuit(
            qubits, arm.get("layers", 2), arm.get("ansatz", "crz-chain")
        )
        if arm.get("frozen", False):
            self.circuit.weights.requires_grad_(False)
        indices = torch.arange(vocabulary_size)
        self.register_buffer("word_bits", ((indices[:, None] >> torch.arange(qubits)) & 1).float())
        mapping = np.arange(vocabulary_size)
        if arm.get("permuted_ids", False):
            mapping = np.random.default_rng(1729).permutation(vocabulary_size)
        self.register_buffer("id_mapping", torch.as_tensor(mapping))
        self.objective = arm.get("loss", "ce")

    def probabilities(self, contexts):
        register = self.circuit.register
        state = register.z.new_zeros(len(contexts), 2**self.circuit.qubits, 2)
        # Group variable-length contexts; never encode padding as a word.
        valid = contexts != self.vocabulary_size
        for length in range(1, contexts.shape[1] + 1):
            selected = valid.sum(-1) == length
            if not bool(selected.any()):
                continue
            words = self.id_mapping[contexts[selected][valid[selected]]].reshape(-1, length)
            # The small microbatch reuses the stable single-context implementation.
            values = np.stack(
                [
                    encoding_angles(
                        context_matrix(ids, self.vocabulary_size, self.context), self.circuit.qubits
                    )
                    for ids in words.cpu().numpy()
                ]
            )
            angles = register.z.new_tensor(values)
            encoded = register.z.new_zeros(len(words), 2**self.circuit.qubits, 2)
            encoded[:, :, 0] = 2 ** (-self.circuit.qubits / 2)
            for layer in angles.unbind(1):
                for q in range(self.circuit.qubits):
                    encoded = register.rotate(encoded, layer[:, q, 0], q, "x")
                encoded = register.entangle(register.phase(encoded, layer[:, :, 1]))
            state[selected] = encoded
        state = self.circuit(state)
        z = state.square().sum(-1) @ register.z
        return ((1 - z) / 2).clamp(1e-7, 1 - 1e-7)

    def forward(self, contexts):
        p = self.probabilities(contexts)
        # Discard invalid bitstrings by normalizing these logits in cross-entropy.
        return p.log() @ self.word_bits.T + torch.log1p(-p) @ (1 - self.word_bits).T

    def training_loss(self, contexts, targets):
        if self.objective == "bce":
            return F.binary_cross_entropy(self.probabilities(contexts), self.word_bits[targets])
        return F.cross_entropy(self(contexts), targets)


class HybridCircuitModel(SemanticModel):
    """Same learned encoder and tied decoder across alternative circuit families."""

    def __init__(self, vocabulary_size, window, arm):
        super().__init__(vocabulary_size, window=window, layers=arm.get("layers", 2))
        # Use one parameter set; remove the inherited original ansatz.
        del self.ansatz
        self.circuit = VariationalCircuit(4, arm.get("layers", 2), arm.get("ansatz", "crz-chain"))
        if arm.get("frozen", False):
            self.circuit.weights.requires_grad_(False)

    def state_from_angles(self, angles):
        state = angles.new_zeros(len(angles), 16, 2)
        state[:, 0, 0] = 1
        for q in range(4):
            state = self.rotate(state, angles[:, q], q, "y")
        return self.circuit(self.phase(state, -0.5 * (angles[:, 4:] @ self.z.T)))


class ClassicalContext(nn.Module):
    """Tied word lookup with ordered MLP or mean-pooling control."""

    def __init__(self, vocabulary_size, window, kind):
        super().__init__()
        self.kind, self.vocabulary_size = kind, vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size + 1, 16, padding_idx=vocabulary_size)
        self.mlp = (
            nn.Identity()
            if kind == "mean"
            else nn.Sequential(nn.Linear(window * 16, 32), nn.GELU(), nn.Linear(32, 16))
        )
        self.bias = nn.Parameter(torch.zeros(vocabulary_size))

    def forward(self, contexts):
        embedded = self.embedding(contexts)
        if self.kind == "mean":
            hidden = embedded.sum(1) / (contexts != self.vocabulary_size).sum(1, keepdim=True)
        else:
            hidden = self.mlp(embedded.flatten(1))
        return F.linear(hidden, self.embedding.weight[:-1], self.bias)


class BertWordModel(nn.Module):
    """Actual BERT MLM architecture, randomly initialized, on shared word IDs."""

    def __init__(self, vocabulary_size, window, objective, arm):
        super().__init__()
        from transformers import BertConfig, BertForMaskedLM

        self.vocabulary_size, self.objective = vocabulary_size, objective
        self.mask_position = window if objective == "causal" else window // 2
        width = arm.get("width", 16)
        config = BertConfig(
            vocab_size=vocabulary_size + 2,  # PAD and MASK never predicted.
            hidden_size=width,
            num_hidden_layers=arm.get("layers", 2),
            num_attention_heads=arm.get("heads", 2),
            intermediate_size=4 * width,
            max_position_embeddings=window + 1,
            pad_token_id=vocabulary_size,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        config._attn_implementation = "eager"
        self.bert = BertForMaskedLM(config)

    def forward(self, contexts):
        mask = torch.full_like(contexts[:, :1], self.vocabulary_size + 1)
        inputs = torch.cat(
            (contexts[:, : self.mask_position], mask, contexts[:, self.mask_position :]), -1
        )
        logits = self.bert(inputs, attention_mask=(inputs != self.vocabulary_size)).logits
        return logits[:, self.mask_position, : self.vocabulary_size]


def create_benchmark_model(vocabulary_size, window, objective, arm):
    kind = arm["model"]
    if kind == "qcse":
        return PaperQCSE(vocabulary_size, arm)
    if kind == "hybrid":
        return HybridCircuitModel(vocabulary_size, window, arm)
    if kind in ("mlp", "mean"):
        return ClassicalContext(vocabulary_size, window, kind)
    if kind == "bert":
        return BertWordModel(vocabulary_size, window, objective, arm)
    if kind == "attention":
        if objective != "causal":
            raise ValueError("Existing quantum attention supports the causal track only")
        return QuantumAttentionModel(
            vocabulary_size,
            window=window,
            layers=arm.get("layers", 2),
            encoding=arm["encoding"],
            circuit=arm["circuit"],
        )
    raise ValueError(f"Unknown model: {kind}")


def training_loss(model, contexts, targets):
    if isinstance(model, PaperQCSE):
        return model.training_loss(contexts, targets)
    return F.cross_entropy(model(contexts), targets)
