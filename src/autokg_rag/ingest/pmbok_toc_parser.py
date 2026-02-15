"""PMBOK Table of Contents parser for hierarchical section detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class TocEntry:
    """Represents a single entry in the PMBOK Table of Contents."""

    level: int
    number: str
    title: str
    page: int
    full_path: str = ""  # Full hierarchical path like "1. Project Management Framework"


class PmbokTocParser:
    """Parses PMBOK PDF to extract hierarchical Table of Contents."""

    def __init__(self) -> None:
        # Patterns for detecting TOC entries in PMBOK format
        # Common patterns for PMBOK TOC entries:
        # "1.1 Project Management Overview .................. 23"
        self.toc_patterns = [
            # Pattern for entries with dots: "1.1.1 Some Title ........ 45"
            re.compile(r'^(\d+(?:\.\d+)*)\s+(.+?)\.{2,}\s+(\d+)$'),
            # Pattern for entries with dots but no dots before page: "1.1.1 Some Title 45"
            re.compile(r'^(\d+(?:\.\d+)*)\s+(.+?)\s+(\d+)$'),
            # Pattern for entries with spaces before page: "1.1.1 Some Title      45"
            re.compile(r'^(\d+(?:\.\d+)*)\s+(.+?)\s{2,}(\d+)$'),
            # Alternative pattern that captures more variations
            re.compile(r'^(\d+(?:\.\d+)*)[.\s]+(.+?)\s+([IVXivx\d]+)$'),  # Roman numerals too
        ]

        # Pattern to identify TOC start in PMBOK
        self.toc_start_patterns = [
            re.compile(r'^Table of Contents$', re.IGNORECASE),
            re.compile(r'^Contents$', re.IGNORECASE),
            re.compile(r'^CONTENTS$', re.IGNORECASE),
        ]

    def parse_toc_from_pdf(self, pdf_path: Path) -> list[TocEntry]:
        """Parse the TOC from a PMBOK PDF file."""
        reader = PdfReader(str(pdf_path))
        page_texts = [(page.extract_text() or "") for page in reader.pages]

        toc_start_index: int | None = None
        for i, text in enumerate(page_texts):
            if self._is_toc_page(text):
                toc_start_index = i
                break

        if toc_start_index is None:
            return []

        # TOC in PMBOK may span multiple pages. Collect a bounded window.
        toc_window = page_texts[toc_start_index : toc_start_index + 30]
        toc_lines: list[str] = []
        for text in toc_window:
            toc_lines.extend(text.splitlines())

        toc_entries = self._extract_toc_from_lines(toc_lines)

        # Build full hierarchical paths
        self._build_hierarchical_paths(toc_entries)

        return toc_entries

    def _is_toc_page(self, text: str | None) -> bool:
        """Check if the page contains TOC headers."""
        if not text:
            return False
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            for pattern in self.toc_start_patterns:
                if pattern.match(line.strip()):
                    return True
        return False

    def _extract_toc_from_lines(self, lines: list[str]) -> list[TocEntry]:
        """Extract TOC entries from TOC lines."""
        entries: list[TocEntry] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try each pattern to match TOC entries
            for pattern in self.toc_patterns:
                match = pattern.search(line)
                if match:
                    number = match.group(1)
                    title = match.group(2).strip()

                    raw_page = match.group(3)
                    page_num = self._parse_page_number(raw_page)
                    if page_num is None:
                        break

                    level = len(number.split('.'))  # Determine nesting level from dots
                    entry = TocEntry(level=level, number=number, title=title, page=page_num)
                    entries.append(entry)
                    break  # Found a match, move to next line

        # Deduplicate by (number, page) while preserving input order.
        dedup: dict[tuple[str, int], TocEntry] = {}
        for entry in entries:
            dedup[(entry.number, entry.page)] = entry

        # Sort by page number to ensure proper order
        return sorted(dedup.values(), key=lambda x: x.page)

    def _parse_page_number(self, raw: str) -> int | None:
        raw = raw.strip()
        if raw.isdigit():
            return int(raw)

        roman = raw.upper()
        roman_map = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
        if any(ch not in roman_map for ch in roman):
            return None

        total = 0
        prev = 0
        for ch in reversed(roman):
            value = roman_map[ch]
            if value < prev:
                total -= value
            else:
                total += value
            prev = value
        return total if total > 0 else None

    def _build_hierarchical_paths(self, entries: list[TocEntry]) -> None:
        """Build full hierarchical paths for each entry."""
        for i, entry in enumerate(entries):
            # Find the parent based on level
            parent_path = ""
            for j in range(i - 1, -1, -1):
                if entries[j].level < entry.level:
                    parent_path = entries[j].full_path
                    break

            # Construct the full path
            if parent_path:
                entry.full_path = f"{parent_path} / {entry.title}"
            else:
                entry.full_path = entry.title

    def create_section_map(
        self,
        toc_entries: list[TocEntry],
        *,
        max_page: int | None = None,
    ) -> dict[int, TocEntry]:
        """Create a mapping from page numbers to their corresponding TOC entries."""
        section_map: dict[int, TocEntry] = {}
        if not toc_entries:
            return section_map

        # Sort entries by page number
        sorted_entries = sorted(toc_entries, key=lambda x: x.page)

        # Create page-to-section mapping
        for i, entry in enumerate(sorted_entries):
            # Determine the end page for this section (until the next section starts)
            end_page = max_page if max_page is not None else entry.page
            if i + 1 < len(sorted_entries):
                end_page = sorted_entries[i + 1].page - 1

            if end_page < entry.page:
                continue

            # Map all pages in this range to this section
            for page_num in range(entry.page, end_page + 1):
                if page_num not in section_map:
                    section_map[page_num] = entry

        return section_map

    def find_section_by_page(
        self,
        section_map: dict[int, TocEntry],
        page_num: int,
    ) -> TocEntry | None:
        """Find the appropriate section for a given page number."""
        if not section_map:
            return None

        # Exact match first
        if page_num in section_map:
            return section_map[page_num]

        # Prefer nearest preceding section when possible.
        lower_or_equal = [p for p in section_map if p <= page_num]
        if lower_or_equal:
            return section_map[max(lower_or_equal)]

        return section_map[min(section_map.keys())]


def load_pmbok_toc(pdf_path: Path) -> tuple[list[TocEntry], dict[int, TocEntry]]:
    """Convenience function to load PMBOK TOC and create section map."""
    parser = PmbokTocParser()
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    toc_entries = parser.parse_toc_from_pdf(pdf_path)
    section_map = parser.create_section_map(toc_entries, max_page=total_pages)
    return toc_entries, section_map
