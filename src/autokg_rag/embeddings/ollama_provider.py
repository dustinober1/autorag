"""Ollama embedding provider with batch-first fallback behavior."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from autokg_rag.embeddings.base import EmbeddingProvider
from autokg_rag.exceptions import RetrievalError
from autokg_rag.ollama.client import OllamaClient


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embed texts via Ollama's embedding APIs."""

    provider_name = "ollama"

    def __init__(
        self,
        *,
        model_name: str,
        client: OllamaClient,
        dim: int = 0,
    ) -> None:
        normalized_model = model_name.strip()
        if not normalized_model:
            raise RetrievalError("Ollama embedding model name must not be empty.")

        if dim < 0:
            raise RetrievalError("Ollama embedding dim must be greater than or equal to zero.")

        self.model_name = normalized_model
        self.client = client
        self.dim = int(dim)
        self._observed_dim = False

    def embed_texts(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Embed texts as a row-aligned float32 matrix."""

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        ordered_texts = list(texts)

        try:
            return self._embed_batch(ordered_texts)
        except RetrievalError as exc:
            if not self._batch_endpoint_unavailable(exc):
                raise

        return self._embed_single(ordered_texts)

    def _embed_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        response = self.client.post_json(
            path="/api/embed",
            payload={
                "model": self.model_name,
                "input": texts,
            },
        )
        if "embeddings" not in response:
            raise RetrievalError("Ollama /api/embed response missing 'embeddings'.")

        matrix = self._coerce_matrix(
            response["embeddings"],
            expected_rows=len(texts),
            source="/api/embed",
        )
        self._record_dim(int(matrix.shape[1]), source="/api/embed")
        return matrix

    def _embed_single(self, texts: list[str]) -> npt.NDArray[np.float32]:
        rows: list[npt.NDArray[np.float32]] = []
        expected_dim: int | None = None

        for text in texts:
            response = self.client.post_json(
                path="/api/embeddings",
                payload={
                    "model": self.model_name,
                    "prompt": text,
                },
            )
            if "embedding" not in response:
                raise RetrievalError("Ollama /api/embeddings response missing 'embedding'.")

            row = self._coerce_row(response["embedding"], source="/api/embeddings")
            if expected_dim is None:
                expected_dim = int(row.shape[0])
            elif int(row.shape[0]) != expected_dim:
                raise RetrievalError(
                    "Ollama /api/embeddings returned inconsistent embedding dimensions."
                )
            rows.append(row)

        matrix = np.vstack(rows).astype(np.float32)
        self._record_dim(int(matrix.shape[1]), source="/api/embeddings")
        return matrix

    def _coerce_matrix(
        self,
        payload: Any,
        *,
        expected_rows: int,
        source: str,
    ) -> npt.NDArray[np.float32]:
        if not isinstance(payload, list):
            raise RetrievalError(f"Ollama {source} 'embeddings' must be a JSON list.")

        if len(payload) != expected_rows:
            raise RetrievalError(
                f"Ollama {source} returned {len(payload)} embeddings for {expected_rows} texts."
            )

        rows: list[npt.NDArray[np.float32]] = []
        expected_dim: int | None = None
        for raw_row in payload:
            row = self._coerce_row(raw_row, source=source)
            if expected_dim is None:
                expected_dim = int(row.shape[0])
            elif int(row.shape[0]) != expected_dim:
                raise RetrievalError(
                    f"Ollama {source} returned inconsistent embedding dimensions."
                )
            rows.append(row)

        matrix = np.vstack(rows).astype(np.float32)
        return matrix

    def _coerce_row(self, payload: Any, *, source: str) -> npt.NDArray[np.float32]:
        if not isinstance(payload, list):
            raise RetrievalError(f"Ollama {source} embedding row must be a JSON list.")

        if not payload:
            raise RetrievalError(f"Ollama {source} embedding row must not be empty.")

        try:
            row = np.asarray(payload, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise RetrievalError(
                f"Ollama {source} embedding row must contain numeric values."
            ) from exc

        if row.ndim != 1:
            raise RetrievalError(f"Ollama {source} embedding row must be one-dimensional.")

        return row.astype(np.float32)

    def _record_dim(self, dim: int, *, source: str) -> None:
        if self._observed_dim and self.dim != dim:
            raise RetrievalError(
                f"Ollama provider dimension changed unexpectedly in {source}: "
                f"expected {self.dim}, got {dim}."
            )

        self.dim = dim
        self._observed_dim = True

    @staticmethod
    def _batch_endpoint_unavailable(exc: RetrievalError) -> bool:
        message = str(exc).lower()
        return (
            "http 404" in message
            or "http 405" in message
            or "http 501" in message
            or ("http 400" in message and "/api/embed" in message)
            or "/api/embed response missing 'embeddings'" in message
        )
