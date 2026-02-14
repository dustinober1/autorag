"""Document discovery and manifest generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autokg_rag.exceptions import IngestError
from autokg_rag.ingest.pdf_parse import discover_pdf_files, parse_pdf_pages_clean, sha256_for_file
from autokg_rag.schemas.records import DocumentManifestRecord


@dataclass
class RawDocument:
    """In-memory representation for a source document."""

    manifest: DocumentManifestRecord
    source_path: Path
    pages: list[str]

def build_raw_documents(input_dir: Path) -> list[RawDocument]:
    """Load source PDFs and produce manifest+page text records."""

    if not input_dir.exists():
        raise IngestError(f"Input path does not exist: {input_dir}")

    pdf_files = discover_pdf_files(input_dir)
    if not pdf_files:
        raise IngestError(f"No PDF files found under {input_dir}")

    docs: list[RawDocument] = []
    for path in pdf_files:
        sha = sha256_for_file(path)
        pages = parse_pdf_pages_clean(path)
        doc_id = f"doc_{sha[:12]}"
        manifest = DocumentManifestRecord(
            doc_id=doc_id,
            source_path=str(path),
            sha256=sha,
            total_pages=len(pages),
        )
        docs.append(RawDocument(manifest=manifest, source_path=path, pages=pages))

    return docs
