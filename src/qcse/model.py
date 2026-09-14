"""Contextual inference and portable checkpoints using exact Qiskit simulation."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from .circuit import DEFAULT_LAYERS, ansatz_circuit, encoding_circuit, measurement_circuit
from .context import ContextConfig, context_matrix, encoding_angles
from .data import make_examples, tokenize


class QCSEModel:
    def __init__(
        self,
        vocabulary,
        layers=DEFAULT_LAYERS,
        context=None,
        window=4,
        objective="causal",
        direction="forward",
        measurement_basis="z",
        seed=42,
    ):
        if len(vocabulary) < 2 or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("At least two unique vocabulary words are required")
        if objective not in ("cbow", "causal"):
            raise ValueError("objective must be 'cbow' or 'causal'")
        if window < 1 or (objective == "cbow" and window % 2):
            raise ValueError("window must be positive; CBOW requires an even total context size")
        self.vocabulary = list(vocabulary)
        self.qubits = (len(vocabulary) - 1).bit_length()
        self.layers, self.window, self.objective, self.direction, self.measurement_basis = (
            layers,
            window,
            objective,
            direction,
            measurement_basis,
        )
        self.context = context or ContextConfig()
        self.ansatz, self.ansatz_parameters = ansatz_circuit(self.qubits, layers, direction)
        self.readout, self.readout_parameters = measurement_circuit(
            self.qubits, measurement_basis
        )
        self.parameters = tuple(self.ansatz_parameters) + tuple(self.readout_parameters)
        ansatz_weights = np.random.default_rng(seed).uniform(
            -np.pi, np.pi, len(self.ansatz_parameters)
        )
        # Every learned-basis model starts as the exact same Z-basis model.
        self.weights = np.r_[ansatz_weights, np.zeros(len(self.readout_parameters))]
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
            raise ValueError("Invalid model weight vector")
        ansatz_values = values[: len(self.ansatz_parameters)]
        return self.ansatz.assign_parameters(
            dict(zip(self.ansatz_parameters, ansatz_values, strict=True))
        )

    def bound_readout(self, weights=None):
        values = self.weights if weights is None else np.asarray(weights)
        if values.shape != (len(self.parameters),) or not np.isfinite(values).all():
            raise ValueError("Invalid model weight vector")
        readout_values = values[len(self.ansatz_parameters) :]
        return self.readout.assign_parameters(
            dict(zip(self.readout_parameters, readout_values, strict=True))
        )

    def circuit(self, context, measured=False):
        circuit = encoding_circuit(self.angles(context), self.direction)
        circuit.compose(self.bound_ansatz(), inplace=True)
        circuit.compose(self.bound_readout(), inplace=True)
        if measured:
            circuit.measure_all()
        return circuit

    def predict_encoded(self, states, weights=None):
        """Return marginal P(q=1), not a distribution over vocabulary words."""
        ansatz = self.bound_ansatz(weights)
        readout = self.bound_readout(weights)
        return np.asarray(
            [
                state.evolve(ansatz).evolve(readout).probabilities() @ self._basis_bits
                for state in states
            ]
        )

    def embed_phrase(self, phrase: str):
        words = tokenize(phrase)
        examples = make_examples([words], self.vocabulary, self.window, self.objective)
        if not examples:
            raise ValueError("A phrase needs at least two known words")
        probabilities = self.predict_encoded([self.encode(e.context) for e in examples])
        result = []
        for e, p in zip(examples, probabilities, strict=True):
            idx = int((p >= 0.5) @ (1 << np.arange(self.qubits)))
            result.append(
                {
                    "word": words[e.position],
                    "position": e.position,
                    "context": [self.vocabulary[i] for i in e.context],
                    "embedding": p.tolist(),
                    "pauli_z": (1 - 2 * p).tolist(),
                    "measurement_basis": self.measurement_basis,
                    "predicted_id": idx,
                    "predicted_word": self.vocabulary[idx] if idx < len(self.vocabulary) else None,
                }
            )
        return result

    def predict_next(self, phrase: str):
        """Predict the next vocabulary word from a causal prefix."""
        if self.objective != "causal":
            raise ValueError("predict_next requires a model trained with objective='causal'")
        words = tokenize(phrase)
        lookup = {word: i for i, word in enumerate(self.vocabulary)}
        try:
            ids = [lookup[word] for word in words]
        except KeyError as error:
            raise ValueError(f"Word outside the saved vocabulary: {error.args[0]!r}") from error
        if not ids:
            raise ValueError("A nonempty phrase is required")
        context = ids[-self.window :]
        probabilities = self.predict_encoded([self.encode(context)])[0]
        predicted_id = int((probabilities >= 0.5) @ (1 << np.arange(self.qubits)))
        return {
            "context": [self.vocabulary[i] for i in context],
            "embedding": probabilities.tolist(),
            "measurement_basis": self.measurement_basis,
            "predicted_id": predicted_id,
            "predicted_word": self.vocabulary[predicted_id]
            if predicted_id < len(self.vocabulary)
            else None,
        }

    def complete_phrase(self, phrase: str, max_new_tokens: int = 1):
        """Greedily append predicted words to a phrase."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        words = tokenize(phrase)
        if not words:
            raise ValueError("A nonempty phrase is required")
        predictions = []
        for _ in range(max_new_tokens):
            prediction = self.predict_next(" ".join(words))
            if prediction["predicted_word"] is None:
                break
            words.append(prediction["predicted_word"])
            predictions.append(prediction)
        return {"phrase": " ".join(words), "predictions": predictions}

    def save(self, path: Path):
        metadata = {
            "format_version": 1,
            "vocabulary": self.vocabulary,
            "layers": self.layers,
            "context": asdict(self.context),
            "window": self.window,
            "objective": self.objective,
            "direction": self.direction,
            "measurement_basis": self.measurement_basis,
        }
        np.savez_compressed(path, metadata=json.dumps(metadata), weights=self.weights)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata"]))
            if metadata.pop("format_version") != 1:
                raise ValueError("Unsupported checkpoint format")
            # Checkpoints written before objective selection were CBOW models.
            metadata.setdefault("objective", "cbow")
            metadata.setdefault("measurement_basis", "z")
            metadata["context"] = ContextConfig(**metadata["context"])
            model = cls(**metadata)
            model.weights = checkpoint["weights"].copy()
            model.bound_ansatz()  # Validate the saved dimensions before inference.
        return model
