from __future__ import annotations

import numpy as np
import pytest

from autokg_rag.embeddings.ollama_provider import OllamaEmbeddingProvider
from autokg_rag.exceptions import RetrievalError


class _BatchClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def post_json(self, *, path: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append((path, payload))
        if path != "/api/embed":
            raise AssertionError(f"Unexpected endpoint: {path}")
        return {"embeddings": [[1.0, 2.0], [3.0, 4.0]]}


class _FallbackClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def post_json(self, *, path: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append((path, payload))
        if path == "/api/embed":
            raise RetrievalError("Ollama request failed with HTTP 404 for /api/embed")

        if path == "/api/embeddings":
            text = str(payload["prompt"])
            if text == "first":
                return {"embedding": [0.1, 0.2, 0.3]}
            if text == "second":
                return {"embedding": [0.4, 0.5, 0.6]}

        raise AssertionError(f"Unexpected call: {path}")


def test_ollama_provider_uses_batch_endpoint_when_available() -> None:
    client = _BatchClient()
    provider = OllamaEmbeddingProvider(model_name="nomic-embed-text", client=client, dim=0)

    matrix = provider.embed_texts(["first", "second"])

    assert matrix.dtype == np.float32
    assert matrix.shape == (2, 2)
    assert provider.dim == 2
    assert [call[0] for call in client.calls] == ["/api/embed"]


def test_ollama_provider_falls_back_to_single_endpoint() -> None:
    client = _FallbackClient()
    provider = OllamaEmbeddingProvider(model_name="nomic-embed-text", client=client, dim=0)

    matrix = provider.embed_texts(["first", "second"])

    assert matrix.dtype == np.float32
    assert matrix.shape == (2, 3)
    assert np.allclose(matrix[0], np.array([0.1, 0.2, 0.3], dtype=np.float32))
    assert np.allclose(matrix[1], np.array([0.4, 0.5, 0.6], dtype=np.float32))
    assert [call[0] for call in client.calls] == [
        "/api/embed",
        "/api/embeddings",
        "/api/embeddings",
    ]


def test_ollama_provider_validates_batch_shape() -> None:
    class _BadBatchClient:
        def post_json(self, *, path: str, payload: dict[str, object]) -> dict[str, object]:
            return {"embeddings": [[1.0, 2.0], [3.0]]}

    provider = OllamaEmbeddingProvider(
        model_name="nomic-embed-text",
        client=_BadBatchClient(),
        dim=0,
    )

    with pytest.raises(RetrievalError, match="inconsistent embedding dimensions"):
        provider.embed_texts(["first", "second"])


def test_ollama_provider_validates_single_fallback_shape() -> None:
    class _BadFallbackClient:
        def __init__(self) -> None:
            self._count = 0

        def post_json(self, *, path: str, payload: dict[str, object]) -> dict[str, object]:
            if path == "/api/embed":
                raise RetrievalError("Ollama request failed with HTTP 404 for /api/embed")

            self._count += 1
            if self._count == 1:
                return {"embedding": [1.0, 2.0]}
            return {"embedding": [1.0, 2.0, 3.0]}

    provider = OllamaEmbeddingProvider(
        model_name="nomic-embed-text",
        client=_BadFallbackClient(),
        dim=0,
    )

    with pytest.raises(RetrievalError, match="inconsistent embedding dimensions"):
        provider.embed_texts(["first", "second"])
