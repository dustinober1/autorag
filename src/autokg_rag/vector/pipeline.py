"""Vector indexing and query pipelines for Milestone 2."""

from __future__ import annotations

import hashlib

from autokg_rag.config import Settings
from autokg_rag.embeddings.base import EmbeddingProvider
from autokg_rag.embeddings.factory import create_embedding_provider
from autokg_rag.embeddings.pipeline import build_embedding_artifacts
from autokg_rag.exceptions import RetrievalError
from autokg_rag.io import read_jsonl_rows, write_jsonl_rows
from autokg_rag.observability import MetricsWriter, StructuredLogger
from autokg_rag.schemas.records import EmbeddingMetaRecord, RetrievalHitRecord
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
            embedding_provider=settings.embedding_provider,
            ollama_base_url=settings.ollama_base_url,
            ollama_timeout_seconds=settings.ollama_timeout_seconds,
        )
        logger.info(
            stage="index_vector",
            event="complete",
            chunks=len(chunks),
            dim=int(embeddings.shape[1]) if embeddings.ndim == 2 else settings.embedding_dim,
            embedding_model=embedding_model,
            embedding_provider=settings.embedding_provider,
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


def _resolve_embedding_metadata(meta: list[EmbeddingMetaRecord]) -> tuple[str, str, int]:
    first = meta[0]
    provider = first.provider.strip().lower()
    model = first.embedding_model.strip()
    dim = int(first.dim)

    for row in meta[1:]:
        if row.provider.strip().lower() != provider:
            raise RetrievalError(
                "Embedding metadata mixes multiple providers. Re-run index-vector."
            )
        if row.embedding_model.strip() != model:
            raise RetrievalError(
                "Embedding metadata mixes multiple embedding models. Re-run index-vector."
            )
        if int(row.dim) != dim:
            raise RetrievalError(
                "Embedding metadata mixes multiple dimensions. Re-run index-vector."
            )

    return provider, model, dim


def resolve_query_embedding_provider(
    *,
    meta: list[EmbeddingMetaRecord],
    settings: Settings,
) -> EmbeddingProvider:
    """Build the embedding provider used for query-time vectorization."""

    meta_provider, meta_model, meta_dim = _resolve_embedding_metadata(meta)
    if settings.embedding_provider != meta_provider:
        raise RetrievalError(
            "Embedding provider mismatch between settings and indexed artifacts "
            f"('{settings.embedding_provider}' != '{meta_provider}'). "
            "Re-run index-vector with the active provider or update AUTORAG_EMBEDDING_PROVIDER."
        )

    if settings.embedding_model != meta_model:
        raise RetrievalError(
            "Embedding model mismatch between settings and indexed artifacts "
            f"('{settings.embedding_model}' != '{meta_model}'). "
            "Re-run index-vector with the active model or update AUTORAG_EMBEDDING_MODEL."
        )

    if meta_provider == "local_hash" and int(settings.embedding_dim) != meta_dim:
        raise RetrievalError(
            "Embedding dimension mismatch for local_hash between settings and indexed artifacts "
            f"({settings.embedding_dim} != {meta_dim}). Re-run index-vector."
        )

    selection = create_embedding_provider(
        embedding_provider=meta_provider,
        embedding_model=meta_model,
        embedding_dim=meta_dim,
        ollama_base_url=settings.ollama_base_url,
        ollama_timeout_seconds=settings.ollama_timeout_seconds,
    )
    return selection.provider


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

        if embeddings.ndim != 2:
            raise RetrievalError("Embedding matrix must be two-dimensional. Re-run index-vector.")

        meta_provider, meta_model, meta_dim = _resolve_embedding_metadata(meta)
        if int(embeddings.shape[1]) != meta_dim:
            raise RetrievalError(
                "Embedding matrix dimension does not match metadata. Re-run index-vector."
            )
        provider = resolve_query_embedding_provider(meta=meta, settings=settings)
        question_id = _question_id(run_id=run_id, question=question)

        hits = retrieve_vector_hits(
            question_id=question_id,
            question=question,
            chunks=ordered_chunks,
            embeddings=embeddings,
            embedding_provider=provider,
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
            embedding_provider=meta_provider,
            embedding_model=meta_model,
        )
        metrics.counter(stage="query_vector", metric_name="hits.count", value=float(len(hits)))

    return hits
