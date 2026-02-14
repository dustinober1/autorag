"""Lexical smoke retriever used before real embedding index integration."""

from __future__ import annotations

import re
from dataclasses import dataclass

from autokg_rag.schemas.records import ChunkRecord

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


@dataclass
class ScoredChunk:
    """Chunk paired with a lexical relevance score."""

    chunk: ChunkRecord
    score: float


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def retrieve_top_chunks(question: str, chunks: list[ChunkRecord], top_k: int) -> list[ScoredChunk]:
    """Rank chunks by Jaccard-like overlap with question tokens."""

    question_tokens = _tokens(question)

    scored: list[ScoredChunk] = []
    for chunk in chunks:
        chunk_tokens = _tokens(chunk.chunk_text)
        denom = max(len(question_tokens | chunk_tokens), 1)
        numer = len(question_tokens & chunk_tokens)
        score = numer / denom
        scored.append(ScoredChunk(chunk=chunk, score=score))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]
