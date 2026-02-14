"""App-facing API service package for Milestone 6."""

from autokg_rag.app_api.doctor import run_demo_doctor
from autokg_rag.app_api.endpoints import (
    add_documents_endpoint,
    check_ollama_health_endpoint,
    create_store_endpoint,
    delete_store_endpoint,
    demo_build_endpoint,
    get_store_info_endpoint,
    ingest_arxiv_endpoint,
    list_documents_endpoint,
    list_ollama_models_endpoint,
    list_runs_endpoint,
    list_stores_endpoint,
    query_endpoint,
    remove_document_endpoint,
    search_arxiv_endpoint,
    upload_documents_endpoint,
)
from autokg_rag.app_api.retention import run_artifact_retention
from autokg_rag.app_api.service import query_service, run_demo_build

__all__ = [
    "add_documents_endpoint",
    "check_ollama_health_endpoint",
    "create_store_endpoint",
    "delete_store_endpoint",
    "demo_build_endpoint",
    "get_store_info_endpoint",
    "ingest_arxiv_endpoint",
    "list_documents_endpoint",
    "list_ollama_models_endpoint",
    "list_runs_endpoint",
    "list_stores_endpoint",
    "query_endpoint",
    "remove_document_endpoint",
    "run_artifact_retention",
    "run_demo_doctor",
    "query_service",
    "run_demo_build",
    "search_arxiv_endpoint",
    "upload_documents_endpoint",
]
