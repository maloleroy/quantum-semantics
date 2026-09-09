"""Contextual inference and portable checkpoints using exact Qiskit simulation."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from .circuit import DEFAULT_LAYERS, ansatz_circuit, encoding_circuit
from .context import ContextConfig, context_matrix, encoding_angles
from .data import make_examples, tokenize


class QCSEModel:
    def __init__(
        self, vocabulary, layers=DEFAULT_LAYERS, context=None, window=4, direction="forward", seed=42
    ):
        if len(vocabulary) < 2 or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("At least two unique vocabulary words are required")
        if window < 2 or window % 2:
            raise ValueError("window must be an even positive context size")
        self.vocabulary = list(vocabulary)
        self.qubits = (len(vocabulary) - 1).bit_length()
        self.layers, self.window, self.direction = layers, window, direction
        self.context = context or ContextConfig()
        self.ansatz, self.parameters = ansatz_circuit(self.qubits, layers, direction)
        self.weights = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(self.parameters))
        self._basis_bits = (
            (np.arange(2**self.qubits)[:, None] >> np.arange(self.qubits)) & 1
        ).astype(float)

    def angles(self, context):
        return encoding_angles(
            context_matrix(context, len(self.vocabulary), self.context), self.qubits
        )

    def encode(self, context):
        return Statevector.from_instruction(encoding_circuit(self.angles(context), self.direction))

    def bound_ansatz(self, weights=None):
        values = self.weights if weights is None else np.asarray(weights)
        if values.shape != (len(self.parameters),) or not np.isfinite(values).all():
            raise ValueError("Invalid ansatz weight vector")
        return self.ansatz.assign_parameters(dict(zip(self.parameters, values)))

    def circuit(self, context, measured=False):
        circuit = encoding_circuit(self.angles(context), self.direction)
        circuit.compose(self.bound_ansatz(), inplace=True)
        if measured:
            circuit.measure_all()
        return circuit

    def predict_encoded(self, states, weights=None):
        """Return marginal P(q=1), not a distribution over vocabulary words."""
        ansatz = self.bound_ansatz(weights)
        return np.asarray(
            [state.evolve(ansatz).probabilities() @ self._basis_bits for state in states]
        )

    def embed_phrase(self, phrase: str):
        words = tokenize(phrase)
        examples = make_examples([words], self.vocabulary, self.window)
        if not examples:
            raise ValueError("A phrase needs at least two known words")
        probabilities = self.predict_encoded([self.encode(e.context) for e in examples])
        result = []
        for e, p in zip(examples, probabilities):
            idx = int((p >= 0.5) @ (1 << np.arange(self.qubits)))
            result.append(
                {
                    "word": words[e.position],
                    "position": e.position,
                    "context": [self.vocabulary[i] for i in e.context],
                    "embedding": p.tolist(),
                    "pauli_z": (1 - 2 * p).tolist(),
                    "predicted_id": idx,
                    "predicted_word": self.vocabulary[idx] if idx < len(self.vocabulary) else None,
                }
            )
        return result

    def save(self, path: Path):
        metadata = {
            "format_version": 1,
            "vocabulary": self.vocabulary,
            "layers": self.layers,
            "context": asdict(self.context),
            "window": self.window,
            "direction": self.direction,
        }
        np.savez_compressed(path, metadata=json.dumps(metadata), weights=self.weights)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata"]))
            if metadata.pop("format_version") != 1:
                raise ValueError("Unsupported checkpoint format")
            metadata["context"] = ContextConfig(**metadata["context"])
            model = cls(**metadata)
            model.weights = checkpoint["weights"].copy()
            model.bound_ansatz()  # Validate the saved dimensions before inference.
        return model
