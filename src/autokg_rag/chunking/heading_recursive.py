"""Heading-recursive chunking wrapper."""

from __future__ import annotations

from autokg_rag.chunking.base import chunk_pages
from autokg_rag.schemas.records import ChunkRecord, PageRecord


def chunk_page(
    page: PageRecord,
    chunk_word_size: int,
    chunk_word_overlap: int,
) -> list[ChunkRecord]:
    """Chunk one page with the heading-recursive strategy."""

    return chunk_pages(
        pages=[page],
        strategy="heading_recursive",
        chunk_word_size=chunk_word_size,
        chunk_word_overlap=chunk_word_overlap,
        sentence_window_size=5,
        semantic_similarity_breakpoint=0.2,
    )
