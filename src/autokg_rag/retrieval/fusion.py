"""Score fusion utilities for hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from autokg_rag.schemas.records import HybridHitRecord, RetrievalHitRecord


@dataclass
class _HybridCandidate:
    """Intermediate representation for a fused chunk candidate."""

    provenance: RetrievalHitRecord
    first_seen: int
    vector_score: float = 0.0
    graph_score: float = 0.0

    @property
    def chunk_id(self) -> str:
        return self.provenance.chunk_id


def _upsert_candidate(
    *,
    source_hits: list[RetrievalHitRecord],
    by_chunk_id: dict[str, _HybridCandidate],
    first_seen: int,
    is_vector: bool,
) -> int:
    for hit in source_hits:
        candidate = by_chunk_id.get(hit.chunk_id)
        if candidate is None:
            candidate = _HybridCandidate(provenance=hit, first_seen=first_seen)
            by_chunk_id[hit.chunk_id] = candidate
            first_seen += 1

        if is_vector:
            candidate.vector_score = max(candidate.vector_score, float(hit.score))
        else:
            candidate.graph_score = max(candidate.graph_score, float(hit.score))

    return first_seen


def fuse_hybrid_hits(
    *,
    question_id: str,
    vector_hits: list[RetrievalHitRecord],
    graph_hits: list[RetrievalHitRecord],
    top_k: int,
    vector_weight: float,
    graph_weight: float,
) -> list[HybridHitRecord]:
    """Fuse vector and graph retrieval scores with deterministic ranking."""

    by_chunk_id: dict[str, _HybridCandidate] = {}

    first_seen = 0
    first_seen = _upsert_candidate(
        source_hits=vector_hits,
        by_chunk_id=by_chunk_id,
        first_seen=first_seen,
        is_vector=True,
    )
    _upsert_candidate(
        source_hits=graph_hits,
        by_chunk_id=by_chunk_id,
        first_seen=first_seen,
        is_vector=False,
    )

    if not by_chunk_id:
        return []

    max_hits = max(1, top_k)
    ranked = sorted(
        by_chunk_id.values(),
        key=lambda candidate: (
            -((vector_weight * candidate.vector_score) + (graph_weight * candidate.graph_score)),
            -candidate.vector_score,
            -candidate.graph_score,
            candidate.first_seen,
        ),
    )

    fused_hits: list[HybridHitRecord] = []
    for rank, candidate in enumerate(ranked[:max_hits], start=1):
        provenance = candidate.provenance
        score = (vector_weight * candidate.vector_score) + (
            graph_weight * candidate.graph_score
        )
        fused_hits.append(
            HybridHitRecord(
                question_id=question_id,
                rank=rank,
                score=float(score),
                vector_score=float(candidate.vector_score),
                graph_score=float(candidate.graph_score),
                chunk_id=provenance.chunk_id,
                doc_id=provenance.doc_id,
                page=provenance.page,
                section=provenance.section,
            )
        )

    return fused_hits
