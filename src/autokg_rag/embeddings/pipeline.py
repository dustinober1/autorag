"""Embedding artifact generation for M2 indexing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from autokg_rag.embeddings.factory import create_embedding_provider
from autokg_rag.exceptions import RetrievalError
from autokg_rag.io import write_parquet_rows
from autokg_rag.schemas.records import ChunkRecord, EmbeddingMetaRecord


def build_embedding_artifacts(
    *,
    artifact_dir: Path,
    chunks: list[ChunkRecord],
    embedding_model: str,
    embedding_dim: int,
    embedding_provider: str = "local_hash",
    ollama_base_url: str = "http://localhost:11434",
    ollama_timeout_seconds: float = 30.0,
) -> npt.NDArray[np.float32]:
    """Build `embeddings.npy` and `embedding_meta.parquet` from chunk records."""

    selection = create_embedding_provider(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        ollama_base_url=ollama_base_url,
        ollama_timeout_seconds=ollama_timeout_seconds,
    )
    provider = selection.provider
    embeddings = provider.embed_texts([chunk.chunk_text for chunk in chunks])

    if embeddings.ndim != 2:
        raise RetrievalError("Embedding provider returned a non-2D embedding matrix.")

    if int(embeddings.shape[0]) != len(chunks):
        raise RetrievalError(
            "Embedding provider returned a row count that does not match chunk count."
        )

    actual_dim = int(embeddings.shape[1])
    actual_model = provider.model_name

    np.save(artifact_dir / "embeddings.npy", embeddings.astype(np.float32))

    meta_rows = [
        EmbeddingMetaRecord(
            chunk_id=chunk.chunk_id,
            row_idx=row_idx,
            provider=selection.provider_name,
            embedding_model=actual_model,
            dim=actual_dim,
        ).model_dump(mode="json")
        for row_idx, chunk in enumerate(chunks)
    ]
    write_parquet_rows(artifact_dir / "embedding_meta.parquet", meta_rows)

    return embeddings
