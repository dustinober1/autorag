from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from autokg_rag.exceptions import RetrievalError
from autokg_rag.schemas.records import EmbeddingMetaRecord
from autokg_rag.vector import pipeline as vector_pipeline


def _meta(*, provider: str, model: str, dim: int) -> list[EmbeddingMetaRecord]:
    return [
        EmbeddingMetaRecord(
            chunk_id="doc_a-p1-c1",
            row_idx=0,
            provider=provider,
            embedding_model=model,
            dim=dim,
        )
    ]


def _settings(**kwargs: Any) -> Any:
    defaults = {
        "embedding_provider": "local_hash",
        "embedding_model": "bge-small-en-v1.5",
        "embedding_dim": 256,
        "ollama_base_url": "http://localhost:11434",
        "ollama_timeout_seconds": 30.0,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_resolve_query_embedding_provider_rejects_provider_mismatch() -> None:
    with pytest.raises(RetrievalError, match="provider mismatch"):
        vector_pipeline.resolve_query_embedding_provider(
            meta=_meta(provider="local_hash", model="bge-small-en-v1.5", dim=256),
            settings=_settings(embedding_provider="ollama", embedding_model="embeddinggemma:300m"),
        )


def test_resolve_query_embedding_provider_rejects_model_mismatch() -> None:
    with pytest.raises(RetrievalError, match="Embedding model mismatch"):
        vector_pipeline.resolve_query_embedding_provider(
            meta=_meta(provider="local_hash", model="bge-small-en-v1.5", dim=256),
            settings=_settings(embedding_model="intfloat/e5-small-v2"),
        )


def test_resolve_query_embedding_provider_uses_indexed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    expected_provider = object()

    def _fake_factory(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(
            provider=expected_provider,
            provider_name=kwargs["embedding_provider"],
        )

    monkeypatch.setattr(vector_pipeline, "create_embedding_provider", _fake_factory)

    provider = vector_pipeline.resolve_query_embedding_provider(
        meta=_meta(provider="ollama", model="embeddinggemma:300m", dim=768),
        settings=_settings(
            embedding_provider="ollama",
            embedding_model="embeddinggemma:300m",
            embedding_dim=256,
        ),
    )

    assert provider is expected_provider
    assert captured["embedding_provider"] == "ollama"
    assert captured["embedding_model"] == "embeddinggemma:300m"
    assert captured["embedding_dim"] == 768
