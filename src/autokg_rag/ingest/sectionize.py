"""Section detection from page text with PMBOK TOC awareness."""

from __future__ import annotations

import re
from pathlib import Path

from .pmbok_toc_parser import TocEntry, load_pmbok_toc

_SECTION_PREFIX_RE = re.compile(r"^(?:section\s+\d+[\.:\-]?\s*)", re.IGNORECASE)

# Global cache for TOC data per document
_toc_cache: dict[str, dict[int, TocEntry]] = {}


def initialize_pmbok_toc_for_document(pdf_path: Path) -> None:
    """Initialize TOC cache for a specific document."""
    doc_id = str(pdf_path.absolute())

    if doc_id not in _toc_cache:
        _, section_map = load_pmbok_toc(pdf_path)
        _toc_cache[doc_id] = section_map


def detect_section(
    page_text: str,
    doc_path: Path | None = None,
    page_num: int | None = None,
) -> str:
    """Detect a best-effort section label from page content, using PMBOK TOC if available."""

    # If we have document path and page number, try to use TOC mapping
    if doc_path and page_num is not None:
        doc_id = str(doc_path.absolute())
        if doc_id in _toc_cache:
            section_entry = _toc_cache[doc_id].get(page_num)
            if section_entry:
                return section_entry.full_path or section_entry.title

    # Fallback to original logic if no TOC available or matching
    for line in page_text.splitlines():
        candidate = line.strip()
        if not candidate or candidate == "(empty page)":
            continue
        cleaned = _SECTION_PREFIX_RE.sub("", candidate).strip(" :-")
        if cleaned:
            return cleaned[:120]

    return "Section"


def resolve_pmbok_section_path(doc_path: Path, page_num: int) -> str:
    """Resolve the full PMBOK section path for a given page number."""
    doc_id = str(doc_path.absolute())

    if doc_id not in _toc_cache:
        _, section_map = load_pmbok_toc(doc_path)
        _toc_cache[doc_id] = section_map

    section_entry = _toc_cache[doc_id].get(page_num)
    if section_entry:
        return section_entry.full_path or section_entry.title

    return f"Page {page_num}"


def get_cross_references(chunk_text: str, current_section_path: str) -> list[str]:
    """Extract potential cross-references from chunk text."""
    # Look for patterns like "Section 1.2", "Chapter 3", "Figure 4.1", etc.
    cross_ref_patterns = [
        r'[Ss]ection\s+(\d+(?:\.\d+)*)',
        r'[Cc]hapter\s+(\d+(?:\.\d+)*)',
        r'[Ff]igure\s+(\d+(?:\.\d+)*)',
        r'[Tt]able\s+(\d+(?:\.\d+)*)',
        r'[Ee]xhibit\s+(\d+(?:\.\d+)*)',
    ]

    refs: list[str] = []
    for pattern in cross_ref_patterns:
        matches = re.findall(pattern, chunk_text)
        for match in matches:
            refs.append(f"Section {match}")  # Standardize format

    # Deduplicate while preserving order.
    _ = current_section_path  # reserved for future contextual filtering
    return list(dict.fromkeys(refs))
