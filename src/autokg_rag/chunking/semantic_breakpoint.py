"""Semantic-breakpoint chunking wrapper."""

from __future__ import annotations

from autokg_rag.chunking.base import chunk_pages
from autokg_rag.schemas.records import ChunkRecord, PageRecord


def chunk_page(
    page: PageRecord,
    chunk_word_size: int,
    semantic_similarity_breakpoint: float,
) -> list[ChunkRecord]:
    """Chunk one page with semantic breakpoints."""

    return chunk_pages(
        pages=[page],
        strategy="semantic_breakpoint",
        chunk_word_size=chunk_word_size,
        chunk_word_overlap=0,
        sentence_window_size=5,
        semantic_similarity_breakpoint=semantic_similarity_breakpoint,
    )
