"""Contextual inference, Qiskit circuits and portable checkpoints."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from .circuit import DEFAULT_LAYERS, ansatz_circuit, encoding_circuit
from .context import ContextConfig, context_matrix, encoding_angles
from .data import make_examples, tokenize
from .simulation import TensorSimulator


class QCSEModel:
    def __init__(
        self,
        vocabulary,
        layers=DEFAULT_LAYERS,
        context=None,
        window=4,
        objective="causal",
        direction="forward",
        seed=42,
        *,
        device="cpu",
        simulation_batch_size=256,
    ):
        if len(vocabulary) < 2 or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("At least two unique vocabulary words are required")
        if objective not in ("cbow", "causal"):
            raise ValueError("objective must be 'cbow' or 'causal'")
        if window < 1 or (objective == "cbow" and window % 2):
            raise ValueError("window must be positive; CBOW requires an even total context size")
        self.vocabulary = list(vocabulary)
        self.qubits = (len(vocabulary) - 1).bit_length()
        self.layers, self.window, self.objective, self.direction = (
            layers,
            window,
            objective,
            direction,
        )
        self.context = context or ContextConfig()
        self.ansatz, self.parameters = ansatz_circuit(self.qubits, layers, direction)
        self.weights = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(self.parameters))
        self.simulator = TensorSimulator(
            self.qubits, layers, direction, device, simulation_batch_size
        )

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
        return self.ansatz.assign_parameters(dict(zip(self.parameters, values, strict=True)))

    def circuit(self, context, measured=False):
        circuit = encoding_circuit(self.angles(context), self.direction)
        circuit.compose(self.bound_ansatz(), inplace=True)
        if measured:
            circuit.measure_all()
        return circuit

    def predict_encoded(self, states, weights=None):
        """Return P(q=1), shaped (contexts, qubits), or (weight sets, contexts, qubits)."""
        return self.simulator.predict(states, self.weights if weights is None else weights)

    def prepare_states(self, states):
        return self.simulator.prepare_states(states)

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
        }
        np.savez_compressed(path, metadata=json.dumps(metadata), weights=self.weights)

    @classmethod
    def load(cls, path: Path, *, device="cpu", simulation_batch_size=256):
        with np.load(path, allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata"]))
            if metadata.pop("format_version") != 1:
                raise ValueError("Unsupported checkpoint format")
            # Checkpoints written before objective selection were CBOW models.
            metadata.setdefault("objective", "cbow")
            metadata["context"] = ContextConfig(**metadata["context"])
            model = cls(**metadata, device=device, simulation_batch_size=simulation_batch_size)
            model.weights = checkpoint["weights"].copy()
            model.bound_ansatz()  # Validate the saved dimensions before inference.
        return model
