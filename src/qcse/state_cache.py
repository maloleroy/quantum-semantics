"""Bound encoded state memory independently of corpus size."""

from collections import OrderedDict

import numpy as np
import torch


class ContextStates:
    """Keep small corpora on device; stream large corpora through a bounded host cache.

    Encoded amplitudes are preserved. The cache budget limits
    retained statevectors, while each temporary simulation batch is separately
    bounded by simulation_batch_size.
    """

    def __init__(self, model, examples, cache_mib=256):
        if cache_mib < 1:
            raise ValueError("state-cache-mib must be positive")
        self.model = model
        unique = dict.fromkeys(e.context for e in examples)
        self.contexts = list(unique)
        lookup = {context: i for i, context in enumerate(self.contexts)}
        self.state_ids = np.array([lookup[e.context] for e in examples], dtype=np.int64)
        # Qiskit retains complex128 states on the host (16 bytes/amplitude).
        self.capacity = max(1, cache_mib * 2**20 // (16 * 2**model.qubits))
        self.cache = OrderedDict()
        self.packed = None
        if len(self.contexts) <= self.capacity:
            if hasattr(model, "encode_many"):
                self.packed = model.encode_many(self.contexts)
            else:
                self.packed = model.prepare_states(
                    [model.encode(context) for context in self.contexts]
                )

    def _states(self, indices):
        if self.packed is not None:
            return self.packed[indices]
        indices = np.asarray(indices, dtype=np.int64)
        requested = {
            int(index): self.cache[int(index)] for index in indices if int(index) in self.cache
        }
        missing = list(dict.fromkeys(int(i) for i in indices if int(i) not in requested))
        if missing:
            contexts = [self.contexts[index] for index in missing]
            encoded = (
                self.model.encode_many(contexts)
                if hasattr(self.model, "encode_many")
                else [self.model.encode(context) for context in contexts]
            )
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.detach().cpu()
            requested.update(
                (index, state.clone() if isinstance(state, torch.Tensor) else state)
                for index, state in zip(missing, encoded, strict=True)
            )
        states = []
        for index in indices:
            index = int(index)
            self.cache[index] = requested[index]
            self.cache.move_to_end(index)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
            states.append(requested[index])
        if states and isinstance(states[0], torch.Tensor):
            return self.model.prepare_states(torch.stack(states))
        return self.model.prepare_states(states)

    def predict(self, example_ids=None, weights=None):
        if example_ids is None and self.packed is not None:
            return self.model.predict_encoded(self.packed, weights)[..., self.state_ids, :]
        # Evaluate repeated contexts once, including for epoch metrics/exports.
        if example_ids is None:
            unique, inverse = np.arange(len(self.contexts)), self.state_ids
        else:
            unique, inverse = np.unique(self.state_ids[example_ids], return_inverse=True)
        count = len(weights) if weights is not None and weights.ndim == 2 else None
        shape = (len(unique), self.model.qubits)
        probabilities = np.empty((count, *shape) if count is not None else shape)
        size = self.model.simulator.batch_size
        for start in range(0, len(unique), size):
            states = self._states(unique[start : start + size])
            probabilities[..., start : start + len(states), :] = self.model.predict_encoded(
                states, weights
            )
        return probabilities[..., inverse, :]
