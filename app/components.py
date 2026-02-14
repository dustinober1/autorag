"""Reusable UI components for the Milestone 6 Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from autokg_rag.schemas.api import AnswerPayload


class StreamlitLike(Protocol):
    """Minimal Streamlit interface used by reusable rendering components."""

    def markdown(self, body: str, *, unsafe_allow_html: bool = False) -> Any: ...

    def subheader(self, body: str) -> Any: ...

    def write(self, body: object) -> Any: ...

    def caption(self, body: str) -> Any: ...

    def json(self, body: object) -> Any: ...

    def dataframe(self, data: object, *, use_container_width: bool = False) -> Any: ...


def apply_styles(st: StreamlitLike, *, css_path: Path) -> None:
    """Apply optional CSS styles when the local stylesheet is present."""

    if not css_path.exists():
        return

    css_text = css_path.read_text(encoding="utf-8").strip()
    if not css_text:
        return
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)


def render_answer_panel(st: StreamlitLike, payload: AnswerPayload) -> None:
    """Render answer summary and sentence-level support details."""

    st.subheader("Answer")
    st.write(payload.answer.answer_text)
    st.caption(f"Citations attached: {len(payload.answer.citations)}")
    st.caption(f"Grounded sentences: {len(payload.citation_trace)}")


def render_citation_panel(st: StreamlitLike, payload: AnswerPayload) -> None:
    """Render citation metadata and supporting retrieval hits."""

    citation_rows = [
        {
            "chunk_id": citation.chunk_id,
            "doc_id": citation.doc_id,
            "page": citation.page,
            "section": citation.section,
        }
        for citation in payload.answer.citations
    ]
    st.dataframe(citation_rows, use_container_width=True)

    hit_rows = [
        {
            "rank": hit.rank,
            "score": round(hit.score, 6),
            "vector_score": round(hit.vector_score, 6),
            "graph_score": round(hit.graph_score, 6),
            "chunk_id": hit.chunk_id,
            "section": hit.section,
        }
        for hit in payload.hits
    ]
    st.caption("Top supporting hits")
    st.dataframe(hit_rows, use_container_width=True)

    trace_rows = [trace.model_dump(mode="json") for trace in payload.citation_trace]
    st.caption("Citation trace")
    st.json(trace_rows)

