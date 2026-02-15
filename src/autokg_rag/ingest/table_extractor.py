"""Lightweight table extraction for PDF pages.

This module intentionally avoids heavyweight optional dependencies so ingest
remains robust in baseline environments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class ExtractedTable:
    """Represents a table-like block extracted from a page."""

    page: int
    table_id: str
    headers: list[str]
    rows: list[list[str]]
    raw_text: str


_PIPE_TABLE_BLOCK_RE = re.compile(r"(?:^|\n)([^\n]*\|[^\n]*(?:\n[^\n]*\|[^\n]*)+)", re.MULTILINE)
_SPACE_ROW_RE = re.compile(r"\S(?:.*?\S)?(?:\s{2,}\S(?:.*?\S)?)+")


def _normalize_cell(cell: str) -> str:
    return re.sub(r"\s+", " ", cell.strip())


def _parse_pipe_table(block: str) -> tuple[list[str], list[list[str]]] | None:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    matrix = [[_normalize_cell(cell) for cell in line.split("|")] for line in lines]
    width = max(len(row) for row in matrix)
    if width < 2:
        return None

    padded = [row + [""] * (width - len(row)) for row in matrix]
    headers = padded[0]
    data = padded[1:]
    if not any(any(cell for cell in row) for row in data):
        return None
    return headers, data


def _parse_space_table(lines: list[str]) -> tuple[list[str], list[list[str]]] | None:
    if len(lines) < 2:
        return None

    split_rows = [re.split(r"\s{2,}", line.strip()) for line in lines]
    width = max(len(row) for row in split_rows)
    if width < 2:
        return None

    padded = [[_normalize_cell(cell) for cell in row] + [""] * (width - len(row)) for row in split_rows]
    headers = padded[0]
    data = padded[1:]
    if not any(any(cell for cell in row) for row in data):
        return None
    return headers, data


def extract_tables_from_pdf(pdf_path: Path) -> list[ExtractedTable]:
    """Extract table-like structures from PDF text.

    Strategy:
    1) Pipe-delimited multi-line blocks
    2) Consecutive lines with multi-space-delimited columns
    """

    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        # Keep ingestion resilient for fixture-like files with .pdf extension.
        return []
    tables: list[ExtractedTable] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        table_counter = 1

        for match in _PIPE_TABLE_BLOCK_RE.finditer(text):
            parsed = _parse_pipe_table(match.group(1))
            if not parsed:
                continue
            headers, rows = parsed
            tables.append(
                ExtractedTable(
                    page=page_idx,
                    table_id=f"p{page_idx}-t{table_counter}",
                    headers=headers,
                    rows=rows,
                    raw_text=match.group(1),
                )
            )
            table_counter += 1

        lines = [line for line in text.splitlines() if _SPACE_ROW_RE.search(line)]
        if len(lines) >= 2:
            parsed = _parse_space_table(lines)
            if parsed:
                headers, rows = parsed
                tables.append(
                    ExtractedTable(
                        page=page_idx,
                        table_id=f"p{page_idx}-t{table_counter}",
                        headers=headers,
                        rows=rows,
                        raw_text="\n".join(lines),
                    )
                )

    return tables


def table_to_markdown(table: ExtractedTable) -> str:
    """Render extracted table as markdown."""
    if not table.headers:
        return ""

    lines = [
        "| " + " | ".join(h.replace("|", "\\|") for h in table.headers) + " |",
        "| " + " | ".join("---" for _ in table.headers) + " |",
    ]
    for row in table.rows:
        cells = [cell.replace("|", "\\|") for cell in row[: len(table.headers)]]
        cells += [""] * max(0, len(table.headers) - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
