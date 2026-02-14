from __future__ import annotations

import pytest
from pydantic import ValidationError

from autokg_rag.schemas.records import ChunkRecord


def test_chunk_requires_doc_page_section_chunk_id() -> None:
    payload = {
        "chunk_id": "doc_a-p1-c1",
        "doc_id": "doc_a",
        "page": 1,
        "section": "Scope",
        "chunk_text": "Scope baseline controls approved changes.",
    }

    for field in ("chunk_id", "doc_id", "page", "section"):
        broken = dict(payload)
        broken.pop(field)
        with pytest.raises(ValidationError):
            ChunkRecord.model_validate(broken)
