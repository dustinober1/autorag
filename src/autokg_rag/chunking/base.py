"""Chunking strategy dispatcher for M2 ingestion."""

from __future__ import annotations

import re
from typing import Literal

from autokg_rag.exceptions import IngestError
from autokg_rag.schemas.records import ChunkRecord, PageRecord

ChunkingStrategy = Literal[
    "fixed",
    "heading_recursive",
    "sentence_window",
    "semantic_breakpoint",
]

SUPPORTED_CHUNKING_STRATEGIES: tuple[ChunkingStrategy, ...] = (
    "fixed",
    "heading_recursive",
    "sentence_window",
    "semantic_breakpoint",
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_pages(
    pages: list[PageRecord],
    strategy: ChunkingStrategy,
    chunk_word_size: int,
    chunk_word_overlap: int,
    sentence_window_size: int,
    semantic_similarity_breakpoint: float,
) -> list[ChunkRecord]:
    """Chunk parsed pages using the selected strategy."""

    if strategy not in SUPPORTED_CHUNKING_STRATEGIES:
        raise IngestError(f"Unsupported chunking strategy: {strategy}")

    chunks: list[ChunkRecord] = []
    for page in pages:
        if strategy == "fixed":
            page_chunks = _chunk_fixed(page, chunk_word_size, chunk_word_overlap, strategy)
        elif strategy == "heading_recursive":
            page_chunks = _chunk_heading_recursive(
                page,
                chunk_word_size,
                chunk_word_overlap,
            )
        elif strategy == "sentence_window":
            page_chunks = _chunk_sentence_window(page, sentence_window_size)
        else:
            page_chunks = _chunk_semantic_breakpoint(
                page,
                chunk_word_size,
                semantic_similarity_breakpoint,
            )
        chunks.extend(page_chunks)

    return chunks


def _split_words(text: str) -> list[str]:
    words = [part for part in text.split() if part]
    return words if words else ["(empty)"]


def _make_chunk(
    *,
    page: PageRecord,
    strategy: str,
    index: int,
    chunk_text: str,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"{page.doc_id}-p{page.page}-{strategy}-c{index}",
        doc_id=page.doc_id,
        page=page.page,
        section=page.section,
        chunk_text=chunk_text.strip() or "(empty)",
    )


def _window_words(words: list[str], size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    windows: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            continue
        windows.append(" ".join(window))
    return windows


def _chunk_fixed(
    page: PageRecord,
    chunk_word_size: int,
    chunk_word_overlap: int,
    strategy_name: str,
) -> list[ChunkRecord]:
    words = _split_words(page.text)
    windows = _window_words(words, chunk_word_size, chunk_word_overlap)
    return [
        _make_chunk(page=page, strategy=strategy_name, index=index, chunk_text=text)
        for index, text in enumerate(windows, start=1)
    ]


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return False
    if stripped.startswith("#"):
        return True
    return stripped == stripped.title() and len(stripped.split()) <= 8


def _split_heading_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_heading_line(line) and current:
            blocks.append(" ".join(current))
            current = [line]
            continue
        current.append(line)

    if current:
        blocks.append(" ".join(current))

    return blocks if blocks else [text]


def _chunk_heading_recursive(
    page: PageRecord,
    chunk_word_size: int,
    chunk_word_overlap: int,
) -> list[ChunkRecord]:
    blocks = _split_heading_blocks(page.text)

    chunks: list[ChunkRecord] = []
    chunk_idx = 1
    for block in blocks:
        block_words = _split_words(block)
        for chunk_text in _window_words(block_words, chunk_word_size, chunk_word_overlap):
            chunks.append(
                _make_chunk(
                    page=page,
                    strategy="heading_recursive",
                    index=chunk_idx,
                    chunk_text=chunk_text,
                )
            )
            chunk_idx += 1

    return chunks


def _split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(normalized) if part.strip()]
    return sentences if sentences else [normalized or "(empty)"]


def _chunk_sentence_window(page: PageRecord, window_size: int) -> list[ChunkRecord]:
    sentences = _split_sentences(page.text)

    chunks: list[ChunkRecord] = []
    for index, start in enumerate(range(0, len(sentences), window_size), start=1):
        window = sentences[start : start + window_size]
        chunks.append(
            _make_chunk(
                page=page,
                strategy="sentence_window",
                index=index,
                chunk_text=" ".join(window),
            )
        )

    return chunks


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[a-zA-Z0-9]+", text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def _chunk_semantic_breakpoint(
    page: PageRecord,
    chunk_word_size: int,
    semantic_similarity_breakpoint: float,
) -> list[ChunkRecord]:
    sentences = _split_sentences(page.text)

    chunks: list[str] = []
    current: list[str] = []
    prev_tokens: set[str] | None = None

    for sentence in sentences:
        tokens = _token_set(sentence)
        current.append(sentence)

        current_word_count = len(" ".join(current).split())
        similarity = 1.0 if prev_tokens is None else _jaccard(prev_tokens, tokens)

        if similarity < semantic_similarity_breakpoint or current_word_count >= chunk_word_size:
            chunks.append(" ".join(current))
            current = []
            prev_tokens = None
            continue

        prev_tokens = tokens

    if current:
        chunks.append(" ".join(current))

    return [
        _make_chunk(page=page, strategy="semantic_breakpoint", index=index, chunk_text=text)
        for index, text in enumerate(chunks, start=1)
    ]
