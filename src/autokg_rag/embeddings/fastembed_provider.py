"""Deterministic local embedding provider used as M2 baseline."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from autokg_rag.embeddings.base import EmbeddingProvider

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """Hash-based embedder with no external model dependency."""

    def __init__(self, model_name: str, dim: int) -> None:
        self.model_name = model_name
        self.dim = dim

    def _embed_one(self, text: str) -> npt.NDArray[np.float32]:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = [token.lower() for token in _TOKEN_RE.findall(text)]

        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], byteorder="big", signed=False) % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += np.float32(sign)

        norm = np.linalg.norm(vector)
        if norm > 0.0:
            vector = vector / norm

        return vector.astype(np.float32)

    def embed_texts(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Embed texts into a row-aligned matrix."""

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        rows = [self._embed_one(text) for text in texts]
        return np.vstack(rows).astype(np.float32)
