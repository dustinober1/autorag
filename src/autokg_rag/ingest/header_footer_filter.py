"""Header/footer filtering helpers for noisy PDF text extraction."""

from __future__ import annotations

import re
from re import Pattern

HEADER_FOOTER_PATTERNS: list[str] = [
    r"^pmbok(?:\u00ae)?\s*guide\b.*$",
    r"^project management institute\.?$",
    r"^pmi(?:\u00ae)?\.?$",
    r"^copyright\s*(?:\u00a9|\(c\))\s*.*$",
    r"^page\s+\d+\s+(?:of|/)\s+\d+$",
    r"^\d+(?:st|nd|rd|th)\s+edition$",
    r"^a guide to the project management body of knowledge\b.*$",
]


def compile_patterns(patterns: list[str]) -> list[Pattern[str]]:
    """Compile case-insensitive regex patterns."""

    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


_FILTER_PATTERNS = compile_patterns(HEADER_FOOTER_PATTERNS)


def is_header_footer_line(line: str) -> bool:
    """Return True when a single line looks like boilerplate."""

    stripped = line.strip()
    if len(stripped) < 2:
        return True

    if stripped.isdigit():
        return True

    return any(pattern.match(stripped) for pattern in _FILTER_PATTERNS)


def remove_header_footer_from_text(text: str) -> str:
    """Drop per-line boilerplate patterns from one page of text."""

    kept_lines: list[str] = []
    for line in text.splitlines():
        if is_header_footer_line(line):
            continue
        kept_lines.append(line.strip())

    compact = "\n".join(line for line in kept_lines if line)
    return compact


def remove_repeated_lines_across_pages(
    pages: list[str],
    *,
    threshold: float = 0.7,
) -> list[str]:
    """Drop lines that appear on many pages (likely headers/footers)."""

    if not pages:
        return []
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0.0, 1.0].")

    total_pages = len(pages)
    if total_pages < 3:
        return [page.strip() or "(empty page)" for page in pages]

    line_occurrences: dict[str, int] = {}
    for page in pages:
        unique_lines = {line.strip() for line in page.splitlines() if line.strip()}
        for line in unique_lines:
            line_occurrences[line] = line_occurrences.get(line, 0) + 1

    lines_to_remove = {
        line
        for line, count in line_occurrences.items()
        if (count / float(total_pages)) >= threshold
    }

    cleaned_pages: list[str] = []
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        cleaned_lines = [line for line in lines if line not in lines_to_remove]
        cleaned = "\n".join(cleaned_lines).strip()
        cleaned_pages.append(cleaned or "(empty page)")

    return cleaned_pages
