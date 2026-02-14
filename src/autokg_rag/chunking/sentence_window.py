"""Sentence-window chunking wrapper."""

from __future__ import annotations

from autokg_rag.chunking.base import chunk_pages
from autokg_rag.schemas.records import ChunkRecord, PageRecord


def chunk_page(page: PageRecord, sentence_window_size: int = 5) -> list[ChunkRecord]:
    """Chunk one page with sentence windows."""

    return chunk_pages(
        pages=[page],
        strategy="sentence_window",
        chunk_word_size=120,
        chunk_word_overlap=20,
        sentence_window_size=sentence_window_size,
        semantic_similarity_breakpoint=0.2,
    )
