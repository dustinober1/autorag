"""Embedding provider protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt


class EmbeddingProvider(Protocol):
    """Protocol for embedding text into fixed-size vectors."""

    model_name: str
    dim: int

    def embed_texts(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Embed an ordered list of texts into a float32 matrix."""
