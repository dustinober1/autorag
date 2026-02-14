"""Section detection from page text."""

from __future__ import annotations

import re

_SECTION_PREFIX_RE = re.compile(r"^(?:section\s+\d+[\.:\-]?\s*)", re.IGNORECASE)


def detect_section(page_text: str) -> str:
    """Detect a best-effort section label from page content."""

    for line in page_text.splitlines():
        candidate = line.strip()
        if not candidate or candidate == "(empty page)":
            continue
        cleaned = _SECTION_PREFIX_RE.sub("", candidate).strip(" :-")
        if cleaned:
            return cleaned[:120]

    return "Section"
