from __future__ import annotations

import pytest
from pydantic import ValidationError

from autokg_rag.schemas.api import AnswerPayload


def test_answer_payload_matches_schema_contract() -> None:
    payload = {
        "answer": {
            "question_id": "m4:q_payload",
            "answer_text": "Scope control affects mitigation planning.",
            "citations": [
                {
                    "chunk_id": "doc_a-p1-c1",
                    "doc_id": "doc_a",
                    "page": 1,
                    "section": "Scope",
                }
            ],
        },
        "hits": [
            {
                "question_id": "m4:q_payload",
                "rank": 1,
                "score": 0.9,
                "vector_score": 0.8,
                "graph_score": 1.0,
                "chunk_id": "doc_a-p1-c1",
                "doc_id": "doc_a",
                "page": 1,
                "section": "Scope",
            }
        ],
        "citation_trace": [
            {
                "answer_sentence_id": "s1",
                "citation": {
                    "chunk_id": "doc_a-p1-c1",
                    "doc_id": "doc_a",
                    "page": 1,
                    "section": "Scope",
                },
                "support_score": 0.9,
            }
        ],
    }

    parsed = AnswerPayload.model_validate(payload)
    assert parsed.answer.question_id == "m4:q_payload"
    assert parsed.hits[0].vector_score == pytest.approx(0.8)
    assert parsed.citation_trace[0].answer_sentence_id == "s1"

    with pytest.raises(ValidationError):
        AnswerPayload.model_validate(
            {
                "answer": payload["answer"],
                "hits": payload["hits"],
            }
        )
