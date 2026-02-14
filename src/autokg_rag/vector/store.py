"""Load and align vector retrieval artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from autokg_rag.exceptions import RetrievalError
from autokg_rag.io import read_parquet_rows
from autokg_rag.schemas.records import ChunkRecord, EmbeddingMetaRecord


def load_chunks(artifact_dir: Path) -> list[ChunkRecord]:
    """Load chunk records from `chunks.parquet`."""

    rows = read_parquet_rows(artifact_dir / "chunks.parquet")
    return [ChunkRecord.model_validate(row) for row in rows]


def load_embedding_meta(artifact_dir: Path) -> list[EmbeddingMetaRecord]:
    """Load embedding metadata from `embedding_meta.parquet`."""

    rows = read_parquet_rows(artifact_dir / "embedding_meta.parquet")
    return [EmbeddingMetaRecord.model_validate(row) for row in rows]


def load_embeddings(artifact_dir: Path) -> npt.NDArray[np.float32]:
    """Load embedding matrix from `embeddings.npy`."""

    path = artifact_dir / "embeddings.npy"
    if not path.exists():
        raise RetrievalError(f"Missing embeddings file: {path}")

    matrix = np.load(path)
    return np.asarray(matrix, dtype=np.float32)


def align_chunks_with_meta(
    chunks: list[ChunkRecord],
    meta_rows: list[EmbeddingMetaRecord],
) -> list[ChunkRecord]:
    """Align chunks to embedding row order via `embedding_meta` records."""

    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}

    ordered: list[tuple[int, ChunkRecord]] = []
    for row in meta_rows:
        chunk = chunk_map.get(row.chunk_id)
        if chunk is None:
            raise RetrievalError(f"Embedding metadata references unknown chunk_id: {row.chunk_id}")
        ordered.append((row.row_idx, chunk))

    ordered.sort(key=lambda item: item[0])
    return [chunk for _, chunk in ordered]
