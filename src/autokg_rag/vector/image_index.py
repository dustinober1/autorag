"""Image caption embedding/index helpers for multimodal retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from autokg_rag.embeddings.base import EmbeddingProvider
from autokg_rag.embeddings.factory import create_embedding_provider
from autokg_rag.io import read_jsonl_rows, write_jsonl_rows
from autokg_rag.vector.index import search_top_k


def create_image_embeddings(
    captions: Sequence[str],
    *,
    model: str = "bge-small-en-v1.5",
    provider: str = "local_hash",
    dim: int = 256,
    ollama_base_url: str = "http://localhost:11434",
    ollama_timeout_seconds: float = 30.0,
) -> list[list[float]]:
    """Create image-caption embeddings using the configured text embedder."""

    selection = create_embedding_provider(
        embedding_provider=provider,
        embedding_model=model,
        embedding_dim=dim,
        ollama_base_url=ollama_base_url,
        ollama_timeout_seconds=ollama_timeout_seconds,
    )
    matrix = selection.provider.embed_texts(list(captions))
    return [[float(value) for value in row] for row in matrix]


@dataclass(frozen=True)
class ImageIndexEntry:
    chunk_id: str
    caption: str
    image_ref: str
    page: int


class ImageIndex:
    """In-memory index for captioned images."""

    def __init__(
        self,
        *,
        embeddings: npt.NDArray[np.float32],
        entries: list[ImageIndexEntry],
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.entries = entries
        self.embedding_provider = embedding_provider

    @classmethod
    def from_captions(
        cls,
        *,
        captions: list[str],
        image_refs: list[str],
        page_nums: list[int],
        embedding_provider: EmbeddingProvider,
    ) -> ImageIndex:
        if not (len(captions) == len(image_refs) == len(page_nums)):
            raise ValueError("captions, image_refs, and page_nums must have equal length.")

        embeddings = embedding_provider.embed_texts(captions)
        entries = [
            ImageIndexEntry(
                chunk_id=f"img_{idx}",
                caption=caption,
                image_ref=image_ref,
                page=int(page),
            )
            for idx, (caption, image_ref, page) in enumerate(
                zip(captions, image_refs, page_nums, strict=True)
            )
        ]
        return cls(embeddings=embeddings, entries=entries, embedding_provider=embedding_provider)

    def save(self, artifact_dir: Path) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        np.save(artifact_dir / "image_embeddings.npy", self.embeddings.astype(np.float32))
        write_jsonl_rows(
            artifact_dir / "image_embedding_meta.jsonl",
            [
                {
                    "row_idx": idx,
                    "chunk_id": entry.chunk_id,
                    "caption": entry.caption,
                    "image_ref": entry.image_ref,
                    "page": entry.page,
                }
                for idx, entry in enumerate(self.entries)
            ],
        )

    @classmethod
    def load(
        cls,
        *,
        artifact_dir: Path,
        embedding_provider: EmbeddingProvider,
    ) -> ImageIndex:
        embeddings = np.asarray(np.load(artifact_dir / "image_embeddings.npy"), dtype=np.float32)
        rows = read_jsonl_rows(artifact_dir / "image_embedding_meta.jsonl")
        entries = [
            ImageIndexEntry(
                chunk_id=str(row.get("chunk_id", f"img_{idx}")),
                caption=str(row.get("caption", "")),
                image_ref=str(row.get("image_ref", "")),
                page=int(row.get("page", 0)),
            )
            for idx, row in enumerate(rows)
        ]
        return cls(embeddings=embeddings, entries=entries, embedding_provider=embedding_provider)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        if not self.entries:
            return []

        query_matrix = self.embedding_provider.embed_texts([query])
        if query_matrix.size == 0:
            return []

        query_vector = np.asarray(query_matrix[0], dtype=np.float32)
        ranked = search_top_k(query_vector=query_vector, embeddings=self.embeddings, top_k=top_k)

        rows: list[dict[str, object]] = []
        for row_idx, score in ranked:
            if row_idx < 0 or row_idx >= len(self.entries):
                continue
            entry = self.entries[row_idx]
            rows.append(
                {
                    "chunk_id": entry.chunk_id,
                    "score": float(score),
                    "caption": entry.caption,
                    "image_ref": entry.image_ref,
                    "page": entry.page,
                }
            )
        return rows
