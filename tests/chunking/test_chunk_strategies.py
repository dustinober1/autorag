from __future__ import annotations

from autokg_rag.chunking.base import SUPPORTED_CHUNKING_STRATEGIES, chunk_pages
from autokg_rag.schemas.records import PageRecord


def test_chunkers_emit_valid_provenance_and_unique_chunk_ids() -> None:
    pages = [
        PageRecord(
            doc_id="doc_alpha",
            page=1,
            section="Project Scope",
            text=(
                "Project scope defines deliverables. "
                "Scope baseline evaluates changes. "
                "Stakeholder alignment tracks approvals."
            ),
        ),
        PageRecord(
            doc_id="doc_alpha",
            page=2,
            section="Risk Response",
            text=(
                "Risk mitigation reduces impact. "
                "Risk acceptance documents residual exposure."
            ),
        ),
    ]

    seen_chunk_ids: set[str] = set()

    for strategy in SUPPORTED_CHUNKING_STRATEGIES:
        chunks = chunk_pages(
            pages=pages,
            strategy=strategy,
            chunk_word_size=20,
            chunk_word_overlap=5,
            sentence_window_size=2,
            semantic_similarity_breakpoint=0.2,
        )

        assert chunks
        for chunk in chunks:
            assert chunk.chunk_id not in seen_chunk_ids
            seen_chunk_ids.add(chunk.chunk_id)
            assert chunk.doc_id == "doc_alpha"
            assert chunk.page in {1, 2}
            assert chunk.section in {"Project Scope", "Risk Response"}
            assert chunk.chunk_text.strip() != ""
