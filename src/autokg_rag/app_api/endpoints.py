"""Endpoint-style wrappers for the Milestone 6 app service."""

from __future__ import annotations

from pathlib import Path

from autokg_rag.app_api.service import list_available_runs, query_service, run_demo_build
from autokg_rag.config import Settings, load_settings
from autokg_rag.schemas.api import AnswerPayload, QueryMode, QueryRequest


def _resolve_settings(settings: Settings | None) -> Settings:
    return load_settings() if settings is None else settings


def list_runs_endpoint(*, settings: Settings | None = None) -> list[str]:
    """Return available run IDs suitable for interactive querying."""

    resolved_settings = _resolve_settings(settings)
    return list_available_runs(settings=resolved_settings)


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

