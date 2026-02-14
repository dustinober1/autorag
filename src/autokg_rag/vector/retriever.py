"""Retrieval helpers for smoke and vector baseline modes."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from autokg_rag.embeddings.base import EmbeddingProvider
from autokg_rag.exceptions import RetrievalError
from autokg_rag.schemas.records import ChunkRecord, RetrievalHitRecord
from autokg_rag.vector.index import search_top_k

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


def retrieve_vector_hits(
    *,
    question_id: str,
    question: str,
    chunks: list[ChunkRecord],
    embeddings: np.ndarray,
    embedding_provider: EmbeddingProvider,
    top_k: int,
) -> list[RetrievalHitRecord]:
    """Retrieve top-k vector hits with provenance fields."""

    if embeddings.ndim != 2:
        raise RetrievalError("Embedding matrix must be two-dimensional.")

    embedding_dim = int(embeddings.shape[1])
    query_matrix = embedding_provider.embed_texts([question])
    if query_matrix.ndim != 2 or int(query_matrix.shape[0]) != 1:
        raise RetrievalError("Embedding provider returned invalid query embedding shape.")
    if int(query_matrix.shape[1]) != embedding_dim:
        raise RetrievalError(
            "Query embedding dimension does not match indexed embeddings. Re-run index-vector."
        )
    query_vector = (
        query_matrix[0]
        if query_matrix.size
        else np.zeros((embedding_dim,), dtype=np.float32)
    )

    ranked = search_top_k(query_vector=query_vector, embeddings=embeddings, top_k=top_k)

    hits: list[RetrievalHitRecord] = []
    for rank, (row_idx, score) in enumerate(ranked, start=1):
        chunk = chunks[row_idx]
        hits.append(
            RetrievalHitRecord(
                question_id=question_id,
                rank=rank,
                score=float(score),
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page=chunk.page,
                section=chunk.section,
            )
        )

    return hits
