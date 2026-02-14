"""arXiv-to-store ingest orchestration."""

from __future__ import annotations

from pathlib import Path

from autokg_rag.app_api.document_service import add_document_paths
from autokg_rag.app_api.store_service import create_store
from autokg_rag.arxiv.client import download_papers
from autokg_rag.config import Settings
from autokg_rag.exceptions import IngestError
from autokg_rag.schemas.api import ArxivPaper, IngestResult


def ingest_arxiv_papers(
    store_name: str,
    papers: list[ArxivPaper],
    settings: Settings,
) -> IngestResult:
    """Download selected arXiv papers and ingest into a target store."""

    normalized_name = store_name.strip()
    if not normalized_name:
        raise IngestError("store_name must not be empty.")
    if not papers:
        raise IngestError("No arXiv papers selected.")

    try:
        create_store(normalized_name, settings)
    except IngestError:
        # Existing stores are valid ingest targets.
        pass

    download_dir = Path("data/raw/pdfs/arxiv")
    downloaded_paths = download_papers(papers, output_dir=download_dir)
    return add_document_paths(normalized_name, downloaded_paths, settings)

