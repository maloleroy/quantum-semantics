"""Quantum context-sensitive embeddings following arXiv:2509.05729v2."""

from .context import ContextConfig, context_matrix, encoding_angles
from .model import QCSEModel

__all__ = ["ContextConfig", "QCSEModel", "context_matrix", "encoding_angles"]
