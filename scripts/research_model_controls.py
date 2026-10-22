"""Small scratch-model prototypes; not registered in the frozen DGX runner.

No pretrained weights/tokenizer/download. Integration needs a versioned runtime,
resume/backend checks and CUDA profiles before production training.
"""

import torch
from torch import nn

from qcse.benchmark_models import HybridCircuitModel


class CausalWordDecoder(nn.Module):
    """A GPT-2 architecture over the same word IDs; score next word at last left word."""

    def __init__(self, vocabulary_size, window=4, width=32, layers=2, heads=2):
        super().__init__()
        from transformers import GPT2Config, GPT2LMHeadModel

        self.vocabulary_size, self.window = vocabulary_size, window
        config = GPT2Config(
            vocab_size=vocabulary_size + 1,
            n_positions=window,
            n_embd=width,
            n_layer=layers,
            n_head=heads,
            pad_token_id=vocabulary_size,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            use_cache=False,
        )
        config.update({"bos_token_id": None, "eos_token_id": None})
        config._attn_implementation = "eager"
        self.gpt = GPT2LMHeadModel(config)

    def sequence_logits(self, contexts):
        if contexts.ndim != 2 or contexts.shape[1] != self.window:
            raise ValueError("Expected fixed-width left contexts")
        # Physical slot positions, matching the existing ordered context protocol.
        positions = torch.arange(self.window, device=contexts.device).expand(len(contexts), -1)
        return self.gpt(
            input_ids=contexts,
            attention_mask=contexts != self.vocabulary_size,
            position_ids=positions,
            use_cache=False,
        ).logits[..., : self.vocabulary_size]

    def forward(self, contexts):
        # Positive target positions always have a real preceding word in the last slot.
        # No labels are passed to the library: avoid its automatic label shift.
        return self.sequence_logits(contexts)[:, -1]


class ClassicalAngleFeatures(HybridCircuitModel):
    """Same lookup/angle encoder/tied head; a bounded classical map replaces the circuit.

    At q8/depth2 the map has 680 parameters versus 46 circuit parameters:
    634 extra (0.34% at vocabulary size 10,864). Feature dimension/bounds match;
    parameter count and quantum Bloch-vector geometry are not exactly matched.
    """

    def __init__(self, vocabulary_size, window=4, qubits=8, layers=2):
        super().__init__(
            vocabulary_size,
            window,
            dict(qubits=qubits, layers=layers, ansatz="crz-chain", init_scale=0.05),
        )
        # Common lookup/encoder/decoder are initialized identically before replacement.
        del self.circuit
        self.feature_map = nn.Sequential(
            nn.Linear(2 * qubits, 2 * qubits),
            nn.GELU(),
            nn.Linear(2 * qubits, 3 * qubits),
            nn.Tanh(),
        )

    def representation(self, contexts):
        return self.decoder(self.feature_map(self.encode(contexts)))
