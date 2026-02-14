"""Application service layer for Milestone 6 demo workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autokg_rag.answer import compose_grounded_answer
from autokg_rag.config import Settings
from autokg_rag.eval.dataset_builder import generate_dataset_from_chunks
from autokg_rag.eval.matrix_runner import run_matrix
from autokg_rag.eval.report import build_experiment_report
from autokg_rag.exceptions import RetrievalError
from autokg_rag.ingest import run_ingest_pipeline
from autokg_rag.io import read_jsonl_rows, write_jsonl_rows
from autokg_rag.kg.pipeline import run_build_kg_pipeline
from autokg_rag.kg.retriever import retrieve_graph_hits
from autokg_rag.ollama import OllamaClient
from autokg_rag.retrieval import run_hybrid_query_pipeline
from autokg_rag.retrieval.ollama_reranker import OllamaReranker
from autokg_rag.retrieval.rerank import RerankResult, with_sequential_ranks
from autokg_rag.schemas.api import AnswerPayload, QueryMode, QueryRequest
from autokg_rag.schemas.records import HybridHitRecord, RetrievalHitRecord
from autokg_rag.vector.pipeline import run_index_vector_pipeline, run_vector_query_pipeline
from autokg_rag.vector.store import load_chunks

_DEFAULT_CHUNKING_STRATEGY = "heading_recursive"
_DEFAULT_DATASET_SIZE = 20
_DEFAULT_MATRIX_TOP_K = 10


def list_available_runs(*, settings: Settings) -> list[str]:
    """List run IDs that contain ingest artifacts usable by the demo app."""

    artifact_root = settings.artifact_root
    if not artifact_root.exists():
        return []

    run_ids: list[str] = []
    for candidate in artifact_root.iterdir():
        if not candidate.is_dir():
            continue
        if (candidate / "chunks.parquet").exists():
            run_ids.append(candidate.name)
    run_ids.sort()
    return run_ids


def _to_hybrid_hit(hit: RetrievalHitRecord, *, mode: QueryMode) -> HybridHitRecord:
    if mode == "vector":
        vector_score = hit.score
        graph_score = 0.0
    elif mode == "graph":
        vector_score = 0.0
        graph_score = hit.score
    else:
        vector_score = hit.score
        graph_score = hit.score

    return HybridHitRecord(
        question_id=hit.question_id,
        rank=hit.rank,
        score=hit.score,
        vector_score=vector_score,
        graph_score=graph_score,
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        page=hit.page,
        section=hit.section,
    )


def _retrieve_hits(
    *,
    request: QueryRequest,
    settings: Settings,
    top_k: int,
) -> list[HybridHitRecord]:
    if request.mode == "hybrid":
        try:
            return run_hybrid_query_pipeline(
                run_id=request.run_id,
                question=request.question,
                top_k=top_k,
                settings=settings,
            )
        except RetrievalError:
            vector_hits = run_vector_query_pipeline(
                run_id=request.run_id,
                question=request.question,
                top_k=top_k,
                settings=settings,
            )
            return [_to_hybrid_hit(hit, mode="vector") for hit in vector_hits]

    artifact_dir = settings.artifact_root / request.run_id
    if request.mode == "vector":
        vector_hits = run_vector_query_pipeline(
            run_id=request.run_id,
            question=request.question,
            top_k=top_k,
            settings=settings,
        )
        return [_to_hybrid_hit(hit, mode="vector") for hit in vector_hits]

    graph_hits = retrieve_graph_hits(
        run_id=request.run_id,
        question=request.question,
        artifact_dir=artifact_dir,
        top_k=top_k,
        max_depth=settings.graph_max_depth,
    )
    return [_to_hybrid_hit(hit, mode="graph") for hit in graph_hits]


def _setting_bool(*, settings: Settings, key: str, default: bool = False) -> bool:
    value = getattr(settings, key, default)
    if isinstance(value, bool):
        return value
    return bool(value)


def _setting_int(*, settings: Settings, key: str, default: int) -> int:
    value = getattr(settings, key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _setting_str(*, settings: Settings, key: str, default: str) -> str:
    value = getattr(settings, key, default)
    if not isinstance(value, str):
        return default
    stripped = value.strip()
    return stripped if stripped else default


def _reranker_enabled(*, settings: Settings) -> bool:
    return _setting_bool(settings=settings, key="reranker_enabled", default=False)


def _reranker_candidate_k(*, settings: Settings, top_k: int) -> int:
    configured = _setting_int(settings=settings, key="reranker_candidate_k", default=top_k)
    return max(top_k, configured)


def _reranker_model(*, settings: Settings) -> str:
    return _setting_str(settings=settings, key="reranker_model", default="llama3:8b")


def _persist_rerank_artifacts(
    *,
    artifact_dir: Path,
    model: str,
    candidate_k: int,
    rerank_result: RerankResult,
    final_hits: list[HybridHitRecord],
) -> None:
    if not final_hits:
        return

    question_id = final_hits[0].question_id
    trace_rows = read_jsonl_rows(artifact_dir / "rerank_trace.jsonl")
    trace_row: dict[str, Any] = {
        "question_id": question_id,
        "model": model,
        "candidate_k": candidate_k,
        "parse_status": rerank_result.parse_status,
    }
    if rerank_result.prompt_hash is not None:
        trace_row["prompt_hash"] = rerank_result.prompt_hash
    if rerank_result.raw_output is not None:
        trace_row["raw_output"] = rerank_result.raw_output
    if rerank_result.error is not None:
        trace_row["error"] = rerank_result.error
    trace_rows.append(trace_row)
    write_jsonl_rows(artifact_dir / "rerank_trace.jsonl", trace_rows)

    reranked_rows = read_jsonl_rows(artifact_dir / "reranked_hits.jsonl")
    reranked_rows.extend(hit.model_dump(mode="json") for hit in final_hits)
    write_jsonl_rows(artifact_dir / "reranked_hits.jsonl", reranked_rows)


def _truncate_hits(*, hits: list[HybridHitRecord], top_k: int) -> list[HybridHitRecord]:
    return with_sequential_ranks(list(hits[: max(1, top_k)]))


def _persist_answer_payload(payload: AnswerPayload, artifact_dir: Path) -> None:
    existing_answers = read_jsonl_rows(artifact_dir / "answers.jsonl")
    write_jsonl_rows(
        artifact_dir / "answers.jsonl",
        existing_answers + [payload.answer.model_dump(mode="json")],
    )

    existing_citation_trace = read_jsonl_rows(artifact_dir / "citation_trace.jsonl")
    write_jsonl_rows(
        artifact_dir / "citation_trace.jsonl",
        existing_citation_trace
        + [trace.model_dump(mode="json") for trace in payload.citation_trace],
    )


def _persist_demo_payload_sample(
    *,
    question: str,
    payload: AnswerPayload,
    artifact_dir: Path,
) -> Path:
    output_path = artifact_dir / "demo_payload_samples.jsonl"
    rows = read_jsonl_rows(output_path)
    rows.append(
        {
            "question": question,
            "answer_record": payload.answer.model_dump(mode="json"),
        }
    )
    write_jsonl_rows(output_path, rows)
    return output_path


def query_service(*, request: QueryRequest, settings: Settings) -> AnswerPayload:
    """Query retrieval pipelines and return a grounded answer payload with citations."""

    artifact_dir = settings.artifact_root / request.run_id
    chunks = load_chunks(artifact_dir)
    if not chunks:
        raise RetrievalError(
            f"No chunks found for run_id '{request.run_id}'. Run ingest before querying."
        )

    reranker_is_enabled = _reranker_enabled(settings=settings)
    retrieval_top_k = (
        _reranker_candidate_k(settings=settings, top_k=request.top_k)
        if reranker_is_enabled
        else request.top_k
    )

    hits = _retrieve_hits(request=request, settings=settings, top_k=retrieval_top_k)
    if not hits:
        raise RetrievalError("Query returned no hits.")

    if reranker_is_enabled:
        model = _reranker_model(settings=settings)
        client = OllamaClient(
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
        reranker = OllamaReranker(model=model, client=client)
        chunk_text_by_id = {chunk.chunk_id: chunk.chunk_text for chunk in chunks}
        rerank_result = reranker.rerank(
            question=request.question,
            hits=hits,
            chunk_text_by_id=chunk_text_by_id,
        )
        hits = _truncate_hits(hits=rerank_result.hits, top_k=request.top_k)
        _persist_rerank_artifacts(
            artifact_dir=artifact_dir,
            model=model,
            candidate_k=retrieval_top_k,
            rerank_result=rerank_result,
            final_hits=hits,
        )

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    answer_record, citation_trace = compose_grounded_answer(
        question=request.question,
        hits=hits,
        chunk_by_id=chunk_by_id,
        max_sentences=3,
    )
    payload = AnswerPayload(
        answer=answer_record,
        hits=hits,
        citation_trace=citation_trace,
    )

    _persist_answer_payload(payload=payload, artifact_dir=artifact_dir)
    _persist_demo_payload_sample(
        question=request.question,
        payload=payload,
        artifact_dir=artifact_dir,
    )
    return payload


def _demo_matrix_config(
    *,
    run_id: str,
    dataset_path: Path,
    reports_dir: Path,
    settings: Settings,
) -> dict[str, Any]:
    reranker_enabled = str(bool(settings.reranker_enabled)).lower()
    return {
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "source_run_id": run_id,
        "reports_dir": str(reports_dir),
        "top_k": _DEFAULT_MATRIX_TOP_K,
        "factors": {
            "chunking": [_DEFAULT_CHUNKING_STRATEGY],
            "embedding_model": [settings.embedding_model],
            "embedding_provider": [settings.embedding_provider],
            "retrieval": ["hybrid"],
            "reranker_enabled": [reranker_enabled],
            "reranker_model": [settings.reranker_model],
        },
    }


def _write_demo_report(
    *,
    run_id: str,
    question: str,
    mode: QueryMode,
    top_k: int,
    answer_payload: AnswerPayload,
    reports_dir: Path,
    matrix_rows: list[dict[str, Any]],
    settings: Settings,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "m6_demo_report.md"

    best_row = matrix_rows[0] if matrix_rows else {}
    key_metrics = {
        "ndcg_at_10": float(best_row.get("ndcg_at_10", 0.0) or 0.0),
        "recall_at_10": float(best_row.get("recall_at_10", 0.0) or 0.0),
        "citation_precision": float(best_row.get("citation_precision", 0.0) or 0.0),
        "faithfulness_proxy": float(best_row.get("faithfulness_proxy", 0.0) or 0.0),
    }

    payload = {
        "run_id": run_id,
        "config": {
            "mode": mode,
            "top_k": top_k,
            "question": question,
            "chunking": _DEFAULT_CHUNKING_STRATEGY,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "reranker_enabled": bool(settings.reranker_enabled),
            "reranker_model": settings.reranker_model,
            "retrieval": "hybrid",
        },
        "key_metrics": key_metrics,
        "screenshots": [
            "Run `uv run streamlit run app/streamlit_app.py` and capture UI screenshots manually."
        ],
    }

    lines = [
        "# Milestone 6 Demo Report",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
        "## Answer Preview",
        "",
        answer_payload.answer.answer_text,
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_demo_build(
    *,
    run_id: str,
    input_dir: Path,
    question: str,
    settings: Settings,
    mode: QueryMode = "hybrid",
    top_k: int = 8,
    reports_dir: Path = Path("reports/milestones"),
    matrix_reports_dir: Path = Path("reports/experiments"),
) -> dict[str, str]:
    """Build a full local demo run and produce milestone report artifacts."""

    run_ingest_pipeline(
        input_dir=input_dir,
        run_id=run_id,
        chunking_strategy=_DEFAULT_CHUNKING_STRATEGY,
        settings=settings,
    )
    run_index_vector_pipeline(
        run_id=run_id,
        embedding_model=settings.embedding_model,
        settings=settings,
    )
    run_build_kg_pipeline(run_id=run_id, settings=settings)

    answer_payload = query_service(
        request=QueryRequest(
            run_id=run_id,
            question=question,
            mode=mode,
            top_k=top_k,
        ),
        settings=settings,
    )

    artifact_dir = settings.artifact_root / run_id
    dataset_path = generate_dataset_from_chunks(
        run_id=run_id,
        input_artifact_dir=artifact_dir,
        target_size=_DEFAULT_DATASET_SIZE,
        output_artifact_root=settings.artifact_root,
    )
    matrix_rows = run_matrix(
        run_id=run_id,
        settings=settings,
        config=_demo_matrix_config(
            run_id=run_id,
            dataset_path=dataset_path,
            reports_dir=matrix_reports_dir,
            settings=settings,
        ),
    )
    build_experiment_report(run_id=run_id, reports_dir=matrix_reports_dir)
    report_path = _write_demo_report(
        run_id=run_id,
        question=question,
        mode=mode,
        top_k=top_k,
        answer_payload=answer_payload,
        reports_dir=reports_dir,
        matrix_rows=matrix_rows,
        settings=settings,
    )

    return {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
        "payload_samples_path": str(artifact_dir / "demo_payload_samples.jsonl"),
        "report_path": str(report_path),
    }
