"""Document discovery and manifest generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from autokg_rag.exceptions import IngestError
from autokg_rag.schemas.records import DocumentManifestRecord


@dataclass
class RawDocument:
    """In-memory representation for a source document."""

    manifest: DocumentManifestRecord
    pages: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8192), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_text(path: Path) -> str:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    return text.strip()


def _split_pages(text: str) -> list[str]:
    pages = [part.strip() for part in text.split("\f") if part.strip()]
    if pages:
        return pages
    return [text.strip()] if text.strip() else ["(empty document)"]


def build_raw_documents(input_dir: Path) -> list[RawDocument]:
    """Load source PDFs and produce manifest+page text records."""

    if not input_dir.exists():
        raise IngestError(f"Input path does not exist: {input_dir}")

    pdf_files = sorted(input_dir.rglob("*.pdf"))
    if not pdf_files:
        raise IngestError(f"No PDF files found under {input_dir}")

    docs: list[RawDocument] = []
    for path in pdf_files:
        sha = _sha256(path)
        text = _decode_text(path)
        pages = _split_pages(text)
        doc_id = f"doc_{sha[:12]}"
        manifest = DocumentManifestRecord(
            doc_id=doc_id,
            source_path=str(path),
            sha256=sha,
            total_pages=len(pages),
        )
        docs.append(RawDocument(manifest=manifest, pages=pages))

    return docs
