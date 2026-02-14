"""Embedding artifact generation for M2 indexing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from autokg_rag.embeddings.fastembed_provider import LocalHashEmbeddingProvider
from autokg_rag.io import write_parquet_rows
from autokg_rag.schemas.records import ChunkRecord, EmbeddingMetaRecord


def build_embedding_artifacts(
    *,
    artifact_dir: Path,
    chunks: list[ChunkRecord],
    embedding_model: str,
    embedding_dim: int,
) -> npt.NDArray[np.float32]:
    """Build `embeddings.npy` and `embedding_meta.parquet` from chunk records."""

    provider = LocalHashEmbeddingProvider(model_name=embedding_model, dim=embedding_dim)
    embeddings = provider.embed_texts([chunk.chunk_text for chunk in chunks])

    np.save(artifact_dir / "embeddings.npy", embeddings.astype(np.float32))

    meta_rows = [
        EmbeddingMetaRecord(
            chunk_id=chunk.chunk_id,
            row_idx=row_idx,
            embedding_model=embedding_model,
            dim=embedding_dim,
        ).model_dump(mode="json")
        for row_idx, chunk in enumerate(chunks)
    ]
    write_parquet_rows(artifact_dir / "embedding_meta.parquet", meta_rows)

    return embeddings
