from __future__ import annotations

import json

from autokg_rag.retrieval.ollama_reranker import OllamaReranker
from autokg_rag.schemas.records import RetrievalHitRecord


class _FakeClient:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        stream: bool = False,
        format: str | None = None,
    ) -> dict[str, str]:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "format": format,
            }
        )
        return {"response": self.response_text}


def _hit(*, chunk_id: str, rank: int) -> RetrievalHitRecord:
    return RetrievalHitRecord(
        question_id="m6:q_rerank",
        rank=rank,
        score=1.0 / float(rank),
        chunk_id=chunk_id,
        doc_id="doc_a",
        page=rank,
        section=f"section_{rank}",
    )


def test_ollama_reranker_reorders_hits_from_valid_json() -> None:
    hits = [
        _hit(chunk_id="doc_a-p1-c1", rank=1),
        _hit(chunk_id="doc_a-p1-c2", rank=2),
        _hit(chunk_id="doc_a-p1-c3", rank=3),
    ]
    client = _FakeClient(
        response_text=json.dumps(
            {"ranked_chunk_ids": ["doc_a-p1-c3", "doc_a-p1-c1", "doc_a-p1-c2"]}
        )
    )
    reranker = OllamaReranker(model="llama3:8b", client=client)
    result = reranker.rerank(
        question="Which chunk is most relevant?",
        hits=hits,
        chunk_text_by_id={
            "doc_a-p1-c1": "Scope details",
            "doc_a-p1-c2": "Cost details",
            "doc_a-p1-c3": "Risk details",
        },
    )

    assert result.parse_status == "ok"
    assert result.prompt_hash
    assert [hit.chunk_id for hit in result.hits] == ["doc_a-p1-c3", "doc_a-p1-c1", "doc_a-p1-c2"]
    assert [hit.rank for hit in result.hits] == [1, 2, 3]
    assert client.calls and client.calls[0]["model"] == "llama3:8b"


def test_ollama_reranker_falls_back_for_invalid_json() -> None:
    hits = [
        _hit(chunk_id="doc_a-p1-c1", rank=1),
        _hit(chunk_id="doc_a-p1-c2", rank=2),
    ]
    client = _FakeClient(response_text="not-json")
    reranker = OllamaReranker(model="llama3:8b", client=client)
    result = reranker.rerank(question="Fallback", hits=hits, chunk_text_by_id={})

    assert result.parse_status == "invalid_json"
    assert [hit.chunk_id for hit in result.hits] == ["doc_a-p1-c1", "doc_a-p1-c2"]
    assert [hit.rank for hit in result.hits] == [1, 2]


def test_ollama_reranker_ignores_unknown_ids_and_appends_remaining() -> None:
    hits = [
        _hit(chunk_id="doc_a-p1-c1", rank=1),
        _hit(chunk_id="doc_a-p1-c2", rank=2),
        _hit(chunk_id="doc_a-p1-c3", rank=3),
    ]
    client = _FakeClient(
        response_text=json.dumps(
            {"ranked_chunk_ids": ["doc_a-p1-c2", "unknown", "doc_a-p1-c2"]}
        )
    )
    reranker = OllamaReranker(model="llama3:8b", client=client)
    result = reranker.rerank(question="Partial IDs", hits=hits, chunk_text_by_id={})

    assert result.parse_status == "partial_ids"
    assert [hit.chunk_id for hit in result.hits] == [
        "doc_a-p1-c2",
        "doc_a-p1-c1",
        "doc_a-p1-c3",
    ]
    assert [hit.rank for hit in result.hits] == [1, 2, 3]
