"""Endpoint-style wrappers for the Milestone 6 app service."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from autokg_rag.app_api.document_service import add_documents, list_documents, remove_document
from autokg_rag.app_api.ollama_model_service import check_ollama_health, list_available_models
from autokg_rag.app_api.service import list_available_runs, query_service, run_demo_build
from autokg_rag.app_api.store_service import (
    create_store,
    delete_store,
    get_store_info,
    list_stores,
)
from autokg_rag.app_api.upload_service import upload_documents
from autokg_rag.arxiv import ingest_arxiv_papers, search_arxiv
from autokg_rag.config import Settings, load_settings
from autokg_rag.schemas.api import (
    AnswerPayload,
    ArxivPaper,
    DocumentInfo,
    IngestResult,
    OllamaModelInfo,
    QueryMode,
    QueryRequest,
    StoreInfo,
)


def _resolve_settings(settings: Settings | None) -> Settings:
    return load_settings() if settings is None else settings


def list_runs_endpoint(*, settings: Settings | None = None) -> list[str]:
    """Return available run IDs suitable for interactive querying."""

    resolved_settings = _resolve_settings(settings)
    return list_available_runs(settings=resolved_settings)


def list_stores_endpoint(*, settings: Settings | None = None) -> list[StoreInfo]:
    """Return all known vector stores under artifact root."""

    resolved_settings = _resolve_settings(settings)
    return list_stores(settings=resolved_settings)


def create_store_endpoint(*, store_name: str, settings: Settings | None = None) -> str:
    """Create one empty store directory and return its canonical name."""

    resolved_settings = _resolve_settings(settings)
    return create_store(store_name, resolved_settings)


def delete_store_endpoint(*, store_name: str, settings: Settings | None = None) -> bool:
    """Delete one store directory."""

    resolved_settings = _resolve_settings(settings)
    return delete_store(store_name, resolved_settings)


def get_store_info_endpoint(*, store_name: str, settings: Settings | None = None) -> StoreInfo:
    """Return detailed metadata for one store."""

    resolved_settings = _resolve_settings(settings)
    return get_store_info(store_name, resolved_settings)


def upload_documents_endpoint(
    *,
    store_name: str,
    files: Sequence[Any],
    settings: Settings | None = None,
) -> IngestResult:
    """Upload and ingest one or more files into a store."""

    resolved_settings = _resolve_settings(settings)
    return upload_documents(store_name, files, resolved_settings)


def list_documents_endpoint(
    *,
    store_name: str,
    settings: Settings | None = None,
) -> list[DocumentInfo]:
    """List document metadata for a store."""

    resolved_settings = _resolve_settings(settings)
    return list_documents(store_name, resolved_settings)


def add_documents_endpoint(
    *,
    store_name: str,
    files: Sequence[Any],
    settings: Settings | None = None,
) -> IngestResult:
    """Incrementally ingest files into an existing store."""

    resolved_settings = _resolve_settings(settings)
    return add_documents(store_name, files, resolved_settings)


def remove_document_endpoint(
    *,
    store_name: str,
    doc_id: str,
    settings: Settings | None = None,
) -> bool:
    """Remove one document from a store and rewrite dependent artifacts."""

    resolved_settings = _resolve_settings(settings)
    return remove_document(store_name, doc_id, resolved_settings)


def list_ollama_models_endpoint(*, settings: Settings | None = None) -> list[OllamaModelInfo]:
    """Return locally available Ollama models."""

    resolved_settings = _resolve_settings(settings)
    return list_available_models(resolved_settings)


def check_ollama_health_endpoint(*, settings: Settings | None = None) -> bool:
    """Return Ollama availability health status."""

    resolved_settings = _resolve_settings(settings)
    return check_ollama_health(resolved_settings)


def search_arxiv_endpoint(*, query: str, max_results: int = 10) -> list[ArxivPaper]:
    """Search arXiv by topic/query string."""

    return search_arxiv(query, max_results=max_results)


def ingest_arxiv_endpoint(
    *,
    store_name: str,
    papers: list[ArxivPaper],
    settings: Settings | None = None,
) -> IngestResult:
    """Download selected arXiv papers and ingest into a store."""

    resolved_settings = _resolve_settings(settings)
    return ingest_arxiv_papers(store_name, papers, resolved_settings)


def query_endpoint(
    *,
    run_id: str,
    question: str,
    mode: QueryMode = "hybrid",
    top_k: int = 8,
    settings: Settings | None = None,
) -> AnswerPayload:
    """Return answer payload for one run/question request."""

    resolved_settings = _resolve_settings(settings)
    request = QueryRequest(
        run_id=run_id,
        question=question,
        mode=mode,
        top_k=top_k,
    )
    return query_service(request=request, settings=resolved_settings)


def demo_build_endpoint(
    *,
    run_id: str,
    input_dir: Path,
    question: str,
    mode: QueryMode = "hybrid",
    top_k: int = 8,
    reports_dir: Path = Path("reports/milestones"),
    matrix_reports_dir: Path = Path("reports/experiments"),
    settings: Settings | None = None,
) -> dict[str, str]:
    """Run the full M6 demo build and return output artifact paths."""

    resolved_settings = _resolve_settings(settings)
    return run_demo_build(
        run_id=run_id,
        input_dir=input_dir,
        question=question,
        settings=resolved_settings,
        mode=mode,
        top_k=top_k,
        reports_dir=reports_dir,
        matrix_reports_dir=matrix_reports_dir,
    )
