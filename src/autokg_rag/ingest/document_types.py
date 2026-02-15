"""Document-type helpers used by ingest and store flows."""

from __future__ import annotations

from pathlib import Path

from autokg_rag.schemas.records import DocumentType


def infer_document_type(source_path: Path | str) -> DocumentType:
    """Infer coarse document type from source path metadata."""

    name = str(source_path).lower()
    if "pmbok" in name:
        return "pmbok"
    return "generic"


def is_pmbok_document(document_type: DocumentType) -> bool:
    """Return whether a document type should use PMBOK-specific logic."""

    return document_type == "pmbok"
