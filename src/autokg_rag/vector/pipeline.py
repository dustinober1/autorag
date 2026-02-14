"""Vector indexing and query pipelines for Milestone 2."""

from __future__ import annotations

import hashlib

from autokg_rag.config import Settings
from autokg_rag.embeddings.pipeline import build_embedding_artifacts
from autokg_rag.exceptions import RetrievalError
from autokg_rag.io import read_jsonl_rows, write_jsonl_rows
from autokg_rag.observability import MetricsWriter, StructuredLogger
from autokg_rag.schemas.records import RetrievalHitRecord
from autokg_rag.vector.retriever import retrieve_vector_hits
from autokg_rag.vector.store import (
    align_chunks_with_meta,
    load_chunks,
    load_embedding_meta,
    load_embeddings,
)


def run_index_vector_pipeline(run_id: str, embedding_model: str, settings: Settings) -> int:
    """Build embedding artifacts for an ingest run."""

    artifact_dir = settings.artifact_root / run_id
    logger = StructuredLogger(run_id=run_id, output_path=artifact_dir / "logs.jsonl")
    metrics = MetricsWriter(run_id=run_id, output_path=artifact_dir / "metrics.jsonl")

    with metrics.timer(stage="index_vector", metric_name="index_vector.seconds"):
        chunks = load_chunks(artifact_dir)
        if not chunks:
            raise RetrievalError("No chunks found. Run ingest before index-vector.")

        embeddings = build_embedding_artifacts(
            artifact_dir=artifact_dir,
            chunks=chunks,
            embedding_model=embedding_model,
            embedding_dim=settings.embedding_dim,
        )
        logger.info(
            stage="index_vector",
            event="complete",
            chunks=len(chunks),
            dim=int(embeddings.shape[1]) if embeddings.ndim == 2 else settings.embedding_dim,
            embedding_model=embedding_model,
        )
        metrics.counter(
            stage="index_vector",
            metric_name="chunks.indexed",
            value=float(len(chunks)),
        )

    return len(chunks)


def _question_id(run_id: str, question: str) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
    return f"{run_id}:q_{digest}"


def run_vector_query_pipeline(
    *,
    run_id: str,
    question: str,
    top_k: int,
    settings: Settings,
) -> list[RetrievalHitRecord]:
    """Execute vector retrieval and persist `vector_hits.jsonl`."""

    artifact_dir = settings.artifact_root / run_id
    logger = StructuredLogger(run_id=run_id, output_path=artifact_dir / "logs.jsonl")
    metrics = MetricsWriter(run_id=run_id, output_path=artifact_dir / "metrics.jsonl")

    with metrics.timer(stage="query_vector", metric_name="query_vector.seconds"):
        chunks = load_chunks(artifact_dir)
        meta = load_embedding_meta(artifact_dir)
        embeddings = load_embeddings(artifact_dir)

        if not chunks or not meta:
            raise RetrievalError("Missing chunks or embedding metadata for vector query.")

        ordered_chunks = align_chunks_with_meta(chunks, meta)
        if int(embeddings.shape[0]) != len(ordered_chunks):
            raise RetrievalError(
                "Embedding rows and chunk metadata count do not match. Re-run index-vector."
            )

        embedding_model = meta[0].embedding_model
        embedding_dim = int(meta[0].dim)
        question_id = _question_id(run_id=run_id, question=question)

        hits = retrieve_vector_hits(
            question_id=question_id,
            question=question,
            chunks=ordered_chunks,
            embeddings=embeddings,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            top_k=top_k,
        )

        previous_rows = read_jsonl_rows(artifact_dir / "vector_hits.jsonl")
        next_rows = previous_rows + [hit.model_dump(mode="json") for hit in hits]
        write_jsonl_rows(artifact_dir / "vector_hits.jsonl", next_rows)

        logger.info(
            stage="query_vector",
            event="complete",
            question_id=question_id,
            hits=len(hits),
            top_k=top_k,
        )
        metrics.counter(stage="query_vector", metric_name="hits.count", value=float(len(hits)))

    return hits
