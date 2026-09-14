"""Bound encoded state memory independently of corpus size."""

from collections import OrderedDict

import numpy as np


class ContextStates:
    """Keep small corpora on device; stream large corpora through a bounded host cache.

    The simulator and Qiskit encoding are unchanged. The cache budget limits
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
            self.packed = model.prepare_states([model.encode(context) for context in self.contexts])

    def _states(self, indices):
        if self.packed is not None:
            return self.packed[indices]
        states = []
        for index in indices:
            index = int(index)
            if index not in self.cache:
                self.cache[index] = self.model.encode(self.contexts[index])
                if len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(index)
            states.append(self.cache[index])
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
