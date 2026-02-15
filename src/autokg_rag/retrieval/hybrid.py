"""Hybrid retrieval pipeline (vector + graph fusion)."""

from __future__ import annotations

import hashlib

from autokg_rag.config import Settings
from autokg_rag.exceptions import RetrievalError
from autokg_rag.io import append_jsonl_rows
from autokg_rag.kg.retriever import retrieve_graph_hits
from autokg_rag.observability import MetricsWriter, StructuredLogger
from autokg_rag.retrieval.fusion import fuse_hybrid_hits
from autokg_rag.schemas.records import HybridHitRecord
from autokg_rag.vector.pipeline import resolve_query_embedding_provider
from autokg_rag.vector.retriever import retrieve_vector_hits
from autokg_rag.vector.store import (
    align_chunks_with_meta,
    load_chunks,
    load_embedding_meta,
    load_embeddings,
)


def _question_id(run_id: str, question: str) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
    return f"{run_id}:q_{digest}"


def run_hybrid_query_pipeline(
    *,
    run_id: str,
    question: str,
    top_k: int,
    settings: Settings,
    vector_weight: float | None = None,
    graph_weight: float | None = None,
) -> list[HybridHitRecord]:
    """Run hybrid retrieval, persist `hybrid_hits.jsonl`, and return fused hits."""

    artifact_dir = settings.artifact_root / run_id
    logger = StructuredLogger(run_id=run_id, output_path=artifact_dir / "logs.jsonl")
    metrics = MetricsWriter(run_id=run_id, output_path=artifact_dir / "metrics.jsonl")

    resolved_vector_weight = (
        settings.hybrid_vector_weight if vector_weight is None else float(vector_weight)
    )
    resolved_graph_weight = (
        settings.hybrid_graph_weight if graph_weight is None else float(graph_weight)
    )

    with metrics.timer(stage="query_hybrid", metric_name="query_hybrid.seconds"):
        chunks = load_chunks(artifact_dir)
        meta = load_embedding_meta(artifact_dir)
        embeddings = load_embeddings(artifact_dir)

        if not chunks or not meta:
            raise RetrievalError("Missing chunks or embedding metadata for hybrid query.")

        ordered_chunks = align_chunks_with_meta(chunks, meta)
        if int(embeddings.shape[0]) != len(ordered_chunks):
            raise RetrievalError(
                "Embedding rows and chunk metadata count do not match. Re-run index-vector."
            )
        if embeddings.ndim != 2:
            raise RetrievalError("Embedding matrix must be two-dimensional. Re-run index-vector.")

        question_id = _question_id(run_id=run_id, question=question)
        provider = resolve_query_embedding_provider(meta=meta, settings=settings)
        vector_hits = retrieve_vector_hits(
            question_id=question_id,
            question=question,
            chunks=ordered_chunks,
            embeddings=embeddings,
            embedding_provider=provider,
            top_k=top_k,
        )
        graph_hits = retrieve_graph_hits(
            run_id=run_id,
            question=question,
            artifact_dir=artifact_dir,
            top_k=top_k,
            max_depth=settings.graph_max_depth,
        )

        fused_hits = fuse_hybrid_hits(
            question_id=question_id,
            vector_hits=vector_hits,
            graph_hits=graph_hits,
            top_k=top_k,
            vector_weight=resolved_vector_weight,
            graph_weight=resolved_graph_weight,
        )

        append_jsonl_rows(
            artifact_dir / "hybrid_hits.jsonl",
            [hit.model_dump(mode="json") for hit in fused_hits],
        )

        logger.info(
            stage="query_hybrid",
            event="complete",
            question_id=question_id,
            hits=len(fused_hits),
            top_k=top_k,
            vector_weight=resolved_vector_weight,
            graph_weight=resolved_graph_weight,
        )
        metrics.counter(
            stage="query_hybrid",
            metric_name="hits.count",
            value=float(len(fused_hits)),
        )

    return fused_hits
