"""arXiv API search and PDF download helpers."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, request

from autokg_rag.exceptions import IngestError
from autokg_rag.schemas.api import ArxivPaper


def _load_arxiv_sdk() -> Any:
    try:
        import arxiv as arxiv_sdk  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - dependency guarded in pyproject.
        raise IngestError(
            "arXiv dependency is not installed. Add `arxiv` to dependencies."
        ) from exc
    return arxiv_sdk


def _extract_arxiv_id(entry_id: str) -> str:
    trimmed = entry_id.strip()
    if not trimmed:
        return ""
    # Example entry URLs:
    # - https://arxiv.org/abs/2401.12345v2
    # - http://arxiv.org/abs/cs/9901001v1
    if "/abs/" in trimmed:
        suffix = trimmed.split("/abs/", maxsplit=1)[1]
        return suffix.strip("/")
    return trimmed.rstrip("/").split("/")[-1]


def _coerce_published(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return datetime.now(tz=UTC)


def search_arxiv(query: str, max_results: int = 10) -> list[ArxivPaper]:
    """Search arXiv by query string."""

    normalized_query = query.strip()
    if not normalized_query:
        raise IngestError("arXiv query must not be empty.")
    if max_results <= 0:
        raise IngestError("max_results must be greater than zero.")

    arxiv_sdk = _load_arxiv_sdk()
    search = arxiv_sdk.Search(
        query=normalized_query,
        max_results=int(max_results),
        sort_by=arxiv_sdk.SortCriterion.Relevance,
    )
    client = arxiv_sdk.Client(page_size=min(max_results, 100), delay_seconds=0, num_retries=2)

    papers: list[ArxivPaper] = []
    for result in client.results(search):
        result_id = _extract_arxiv_id(str(getattr(result, "entry_id", "")))
        if not result_id and hasattr(result, "get_short_id"):
            try:
                result_id = str(result.get_short_id())
            except Exception:  # noqa: BLE001
                result_id = ""
        if not result_id:
            continue

        title = str(getattr(result, "title", "")).strip()
        if not title:
            continue

        abstract = str(getattr(result, "summary", "") or "").strip()
        pdf_url = str(getattr(result, "pdf_url", "") or "").strip()
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{result_id}.pdf"

        raw_authors = getattr(result, "authors", []) or []
        authors = [str(getattr(author, "name", author)).strip() for author in raw_authors]
        authors = [author for author in authors if author]

        papers.append(
            ArxivPaper(
                arxiv_id=result_id,
                title=title,
                authors=authors,
                abstract=abstract,
                pdf_url=pdf_url,
                published=_coerce_published(getattr(result, "published", None)),
            )
        )

    return papers


def _safe_pdf_name(arxiv_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", arxiv_id.strip())
    return f"{normalized or 'paper'}.pdf"


def download_papers(papers: list[ArxivPaper], output_dir: Path) -> list[Path]:
    """Download selected arXiv PDFs to `output_dir`."""

    if not papers:
        raise IngestError("No arXiv papers selected for download.")

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths: list[Path] = []

    for paper in papers:
        output_path = output_dir / _safe_pdf_name(paper.arxiv_id)
        if output_path.exists():
            downloaded_paths.append(output_path)
            continue

        try:
            with request.urlopen(paper.pdf_url, timeout=60.0) as response:
                payload = response.read()
        except error.URLError as exc:
            raise IngestError(f"Failed to download arXiv PDF: {paper.pdf_url}") from exc

        if not payload:
            raise IngestError(f"Downloaded empty PDF payload for {paper.arxiv_id}.")

        output_path.write_bytes(payload)
        downloaded_paths.append(output_path)

    return downloaded_paths
