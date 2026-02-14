"""Upload orchestration for Streamlit multi-file ingest."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from autokg_rag.app_api.document_service import UploadedFileLike, add_documents
from autokg_rag.app_api.store_service import create_store
from autokg_rag.config import Settings
from autokg_rag.exceptions import IngestError
from autokg_rag.schemas.api import IngestResult


def upload_documents(
    store_name: str,
    files: Sequence[UploadedFileLike | Any],
    settings: Settings,
) -> IngestResult:
    """Upload one or more PDF files into a store and index embeddings."""

    normalized_name = store_name.strip()
    if not normalized_name:
        raise IngestError("store_name must not be empty.")
    if not files:
        raise IngestError("No files uploaded.")

    try:
        create_store(normalized_name, settings)
    except IngestError:
        # Existing stores are valid upload targets.
        pass

    return add_documents(normalized_name, files, settings)

