"""Factory helpers for selecting embedding providers."""

from __future__ import annotations

from dataclasses import dataclass

from autokg_rag.embeddings.base import EmbeddingProvider
from autokg_rag.embeddings.fastembed_provider import LocalHashEmbeddingProvider
from autokg_rag.embeddings.ollama_provider import OllamaEmbeddingProvider
from autokg_rag.exceptions import RetrievalError
from autokg_rag.ollama.client import OllamaClient


@dataclass(frozen=True)
class ProviderSelection:
    """Resolved provider identity and implementation instance."""

    provider_name: str
    provider: EmbeddingProvider


def create_embedding_provider(
    *,
    embedding_provider: str,
    embedding_model: str,
    embedding_dim: int,
    ollama_base_url: str = "http://localhost:11434",
    ollama_timeout_seconds: float = 30.0,
) -> ProviderSelection:
    """Build an embedding provider from resolved runtime settings."""

    if embedding_dim <= 0:
        raise RetrievalError(f"Embedding dimension must be positive, got {embedding_dim}.")

    requested = embedding_provider.strip().lower() if embedding_provider else "local_hash"
    normalized = {
        "local": "local_hash",
        "local-hash": "local_hash",
        "local_hash": "local_hash",
        "ollama": "ollama",
    }.get(requested)

    if normalized is None:
        raise RetrievalError(
            "Unsupported embedding provider "
            f"'{embedding_provider}'. Expected one of: local_hash, ollama."
        )

    if normalized == "local_hash":
        provider = LocalHashEmbeddingProvider(model_name=embedding_model, dim=embedding_dim)
        return ProviderSelection(provider_name=normalized, provider=provider)

    client = OllamaClient(
        base_url=ollama_base_url,
        timeout_seconds=float(ollama_timeout_seconds),
    )
    ollama_provider = OllamaEmbeddingProvider(
        model_name=embedding_model,
        client=client,
        dim=embedding_dim,
    )
    return ProviderSelection(provider_name=normalized, provider=ollama_provider)
