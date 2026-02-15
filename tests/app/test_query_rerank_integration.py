from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch

from autokg_rag.app_api import service
from autokg_rag.io import read_jsonl_rows
from autokg_rag.retrieval.rerank import RerankResult
from autokg_rag.schemas.api import QueryRequest
from autokg_rag.schemas.records import ChunkRecord, RetrievalHitRecord


def _chunk(*, idx: int) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"doc_a-p1-c{idx}",
        doc_id="doc_a",
        page=idx,
        section=f"section_{idx}",
        chunk_text=f"Evidence text for chunk {idx}.",
    )


def _vector_hit(*, question_id: str, chunk: ChunkRecord, rank: int) -> RetrievalHitRecord:
    return RetrievalHitRecord(
        question_id=question_id,
        rank=rank,
        score=1.0 / float(rank),
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        page=chunk.page,
        section=chunk.section,
    )


def test_query_service_reranks_candidates_and_truncates_to_top_k(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    run_id = "m6"
    artifact_root = tmp_path / "artifacts"
    chunks = [_chunk(idx=1), _chunk(idx=2), _chunk(idx=3), _chunk(idx=4)]

    monkeypatch.setattr(service, "load_chunks", lambda _artifact_dir: chunks)

    observed: dict[str, int] = {}

    def _fake_vector_query(
        *,
        run_id: str,
        question: str,
        top_k: int,
        settings: object,
    ) -> list[RetrievalHitRecord]:
        observed["top_k"] = top_k
        question_id = f"{run_id}:q_test"
        return [
            _vector_hit(question_id=question_id, chunk=chunk, rank=idx)
            for idx, chunk in enumerate(chunks[:top_k], start=1)
        ]

    monkeypatch.setattr(service, "run_vector_query_pipeline", _fake_vector_query)

    class _FakeReranker:
        def __init__(self, *, model: str, client: object) -> None:
            self.model = model
            self.client = client

        def rerank(
            self,
            *,
            question: str,
            hits: list[object],
            chunk_text_by_id: dict[str, str],
        ) -> RerankResult:
            ordered = [hits[2], hits[0], hits[1], hits[3]]
            reranked = [
                hit.model_copy(update={"rank": idx})
                for idx, hit in enumerate(ordered, start=1)
            ]
            return RerankResult(
                hits=reranked,
                parse_status="ok",
                prompt_hash="abc123",
                raw_output=json.dumps(
                    {
                        "ranked_chunk_ids": [
                            hits[2].chunk_id,
                            hits[0].chunk_id,
                            hits[1].chunk_id,
                            hits[3].chunk_id,
                        ]
                    }
                ),
            )

    monkeypatch.setattr(service, "OllamaReranker", _FakeReranker)

    settings = SimpleNamespace(
        artifact_root=artifact_root,
        graph_max_depth=2,
        reranker_enabled=True,
        reranker_candidate_k=4,
        reranker_model="llama3:8b",
        ollama_base_url="http://localhost:11434",
        ollama_timeout_seconds=10.0,
        ollama_api_key="",
    )
    payload = service.query_service(
        request=QueryRequest(
            run_id=run_id,
            question="What is most relevant?",
            mode="vector",
            top_k=2,
        ),
        settings=settings,
    )

    assert observed["top_k"] == 4
    assert [hit.chunk_id for hit in payload.hits] == ["doc_a-p1-c3", "doc_a-p1-c1"]
    assert [hit.rank for hit in payload.hits] == [1, 2]
    assert all(hit.doc_id == "doc_a" for hit in payload.hits)

    trace_rows = read_jsonl_rows(artifact_root / run_id / "rerank_trace.jsonl")
    assert trace_rows
    assert trace_rows[-1]["model"] == "llama3:8b"
    assert trace_rows[-1]["candidate_k"] == 4
    assert trace_rows[-1]["parse_status"] == "ok"
    assert trace_rows[-1]["prompt_hash"]

    reranked_rows = read_jsonl_rows(artifact_root / run_id / "reranked_hits.jsonl")
    assert [row["chunk_id"] for row in reranked_rows[-2:]] == ["doc_a-p1-c3", "doc_a-p1-c1"]


def test_query_service_defaults_to_no_rerank_when_flag_missing(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    run_id = "m6"
    artifact_root = tmp_path / "artifacts"
    chunks = [_chunk(idx=1), _chunk(idx=2), _chunk(idx=3)]

    monkeypatch.setattr(service, "load_chunks", lambda _artifact_dir: chunks)

    observed: dict[str, int] = {}

    def _fake_vector_query(
        *,
        run_id: str,
        question: str,
        top_k: int,
        settings: object,
    ) -> list[RetrievalHitRecord]:
        observed["top_k"] = top_k
        question_id = f"{run_id}:q_test"
        return [
            _vector_hit(question_id=question_id, chunk=chunk, rank=idx)
            for idx, chunk in enumerate(chunks[:top_k], start=1)
        ]

    monkeypatch.setattr(service, "run_vector_query_pipeline", _fake_vector_query)

    def _unexpected_reranker(*args: object, **kwargs: object) -> object:
        raise AssertionError("Reranker should not be constructed when disabled by default.")

    monkeypatch.setattr(service, "OllamaReranker", _unexpected_reranker)

    settings = SimpleNamespace(
        artifact_root=artifact_root,
        graph_max_depth=2,
    )
    payload = service.query_service(
        request=QueryRequest(
            run_id=run_id,
            question="Fallback mode",
            mode="vector",
            top_k=2,
        ),
        settings=settings,
    )

    assert observed["top_k"] == 2
    assert [hit.chunk_id for hit in payload.hits] == ["doc_a-p1-c1", "doc_a-p1-c2"]

    assert not (artifact_root / run_id / "rerank_trace.jsonl").exists()
    assert not (artifact_root / run_id / "reranked_hits.jsonl").exists()
