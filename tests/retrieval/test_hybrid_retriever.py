from __future__ import annotations

import pytest

from autokg_rag.retrieval.fusion import fuse_hybrid_hits
from autokg_rag.schemas.records import RetrievalHitRecord


def _hit(*, question_id: str, chunk_id: str, score: float, rank: int) -> RetrievalHitRecord:
    return RetrievalHitRecord(
        question_id=question_id,
        rank=rank,
        score=score,
        chunk_id=chunk_id,
        doc_id="doc_a",
        page=1,
        section="Scope",
    )


def test_hybrid_fusion_combines_vector_and_graph_scores() -> None:
    question_id = "m4:q_hybrid"
    vector_hits = [
        _hit(question_id=question_id, chunk_id="doc_a-p1-c1", score=0.8, rank=1),
        _hit(question_id=question_id, chunk_id="doc_a-p1-c2", score=0.7, rank=2),
    ]
    graph_hits = [
        _hit(question_id=question_id, chunk_id="doc_a-p1-c1", score=0.4, rank=1),
        _hit(question_id=question_id, chunk_id="doc_a-p2-c1", score=0.9, rank=2),
    ]

    fused = fuse_hybrid_hits(
        question_id=question_id,
        vector_hits=vector_hits,
        graph_hits=graph_hits,
        top_k=5,
        vector_weight=0.6,
        graph_weight=0.4,
    )

    assert fused
    assert fused[0].chunk_id == "doc_a-p1-c1"
    assert [hit.rank for hit in fused] == list(range(1, len(fused) + 1))

    by_chunk = {hit.chunk_id: hit for hit in fused}
    chunk = by_chunk["doc_a-p1-c1"]
    assert chunk.vector_score == pytest.approx(0.8)
    assert chunk.graph_score == pytest.approx(0.4)
    assert chunk.score == pytest.approx((0.6 * 0.8) + (0.4 * 0.4))
