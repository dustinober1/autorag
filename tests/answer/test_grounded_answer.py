from __future__ import annotations

from autokg_rag.answer.composer import compose_grounded_answer
from autokg_rag.answer.grounding import split_sentences
from autokg_rag.schemas.records import ChunkRecord, HybridHitRecord


def test_each_answer_sentence_has_supporting_citation() -> None:
    hits = [
        HybridHitRecord(
            question_id="m4:q_grounded",
            rank=1,
            score=0.92,
            vector_score=0.85,
            graph_score=0.99,
            chunk_id="doc_a-p1-c1",
            doc_id="doc_a",
            page=1,
            section="Scope",
        ),
        HybridHitRecord(
            question_id="m4:q_grounded",
            rank=2,
            score=0.73,
            vector_score=0.61,
            graph_score=0.85,
            chunk_id="doc_a-p2-c1",
            doc_id="doc_a",
            page=2,
            section="Risk",
        ),
    ]
    chunk_by_id = {
        "doc_a-p1-c1": ChunkRecord(
            chunk_id="doc_a-p1-c1",
            doc_id="doc_a",
            page=1,
            section="Scope",
            chunk_text="Scope control affects mitigation planning.",
        ),
        "doc_a-p2-c1": ChunkRecord(
            chunk_id="doc_a-p2-c1",
            doc_id="doc_a",
            page=2,
            section="Risk",
            chunk_text="Mitigation planning influences risk response decisions.",
        ),
    }

    answer, citation_trace = compose_grounded_answer(
        question="How does scope control affect risk response?",
        hits=hits,
        chunk_by_id=chunk_by_id,
        max_sentences=3,
    )

    answer_sentences = split_sentences(answer.answer_text)
    assert answer_sentences

    expected_sentence_ids = {f"s{idx}" for idx in range(1, len(answer_sentences) + 1)}
    trace_sentence_ids = {trace.answer_sentence_id for trace in citation_trace}

    assert expected_sentence_ids <= trace_sentence_ids
    assert answer.citations

    citation_chunk_ids = {citation.chunk_id for citation in answer.citations}
    for trace in citation_trace:
        assert trace.citation.chunk_id in citation_chunk_ids
        assert trace.support_score > 0.0
