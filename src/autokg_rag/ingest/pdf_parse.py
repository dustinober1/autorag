"""PDF parsing utilities with deterministic fallback behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pypdf import PdfReader

from autokg_rag.exceptions import IngestError
from autokg_rag.ingest.header_footer_filter import (
    remove_header_footer_from_text,
    remove_repeated_lines_across_pages,
)
from .table_extractor import ExtractedTable, extract_tables_from_pdf


def sha256_for_file(path: Path) -> str:
    """Return stable SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8192), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_pdf_files(input_dir: Path) -> list[Path]:
    """Discover PDF files under an input directory in stable order."""

    if not input_dir.exists():
        raise IngestError(f"Input path does not exist: {input_dir}")

    pdf_files = sorted(input_dir.rglob("*.pdf"))
    if not pdf_files:
        raise IngestError(f"No PDF files found under {input_dir}")

    return pdf_files


def _normalize_page_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    compact = "\n".join(line for line in lines if line)
    cleaned = remove_header_footer_from_text(compact)
    return cleaned.strip() or "(empty page)"


def _parse_real_pdf(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        pages.append(_normalize_page_text(extracted))
    return pages


def _parse_text_fallback(path: Path) -> list[str]:
    raw = path.read_bytes()
    decoded = raw.decode("utf-8", errors="ignore")
    parts = [part for part in decoded.split("\f") if part.strip()]
    if not parts and decoded.strip():
        parts = [decoded]
    if not parts:
        return ["(empty page)"]
    return [_normalize_page_text(part) for part in parts]


def parse_pdf_pages(path: Path) -> list[str]:
    """Extract page text from a PDF, or fallback to UTF-8 text parsing."""

    raw = path.read_bytes()
    if raw.startswith(b"%PDF"):
        try:
            parsed_pages = _parse_real_pdf(path)
            if parsed_pages:
                return parsed_pages
        except Exception:
            # Some fixture files may have .pdf extension without valid PDF bytes.
            pass

    return _parse_text_fallback(path)


def parse_pdf_pages_clean(
    path: Path,
    *,
    repeated_line_threshold: float = 0.7,
) -> list[str]:
    """Extract pages, then remove lines repeated across most pages."""

    pages = parse_pdf_pages(path)
    cleaned_pages = remove_repeated_lines_across_pages(
        pages,
        threshold=repeated_line_threshold,
    )
    return [page.strip() or "(empty page)" for page in cleaned_pages]


def extract_title(pages: list[str], fallback: str) -> str:
    """Derive a lightweight title from the first non-empty line."""

    if not pages:
        return fallback

    for line in pages[0].splitlines():
        stripped = line.strip()
        if stripped and stripped != "(empty page)":
            return stripped[:200]

    return fallback


def parse_pdf_with_tables(path: Path) -> tuple[list[str], list[ExtractedTable]]:
    """Extract page text and tables from a PDF."""
    pages = parse_pdf_pages_clean(path)
    tables = extract_tables_from_pdf(path)
    return pages, tables
