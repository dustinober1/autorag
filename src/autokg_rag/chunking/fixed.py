"""Simple fixed-size word chunking for smoke pipeline."""

from __future__ import annotations

from autokg_rag.schemas.records import ChunkRecord


def chunk_page(
    *,
    doc_id: str,
    page: int,
    section: str,
    text: str,
    chunk_word_size: int,
    chunk_word_overlap: int,
) -> list[ChunkRecord]:
    """Split page text into deterministic fixed-size chunks."""

    words = text.split()
    if not words:
        words = ["(empty)"]

    step = max(1, chunk_word_size - chunk_word_overlap)
    chunks: list[ChunkRecord] = []

    for idx, start in enumerate(range(0, len(words), step), start=1):
        chunk_words = words[start : start + chunk_word_size]
        if not chunk_words:
            continue
        chunk_text = " ".join(chunk_words)
        chunk_id = f"{doc_id}-p{page}-c{idx}"
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc_id,
                page=page,
                section=section,
                chunk_text=chunk_text,
            )
        )

    return chunks
