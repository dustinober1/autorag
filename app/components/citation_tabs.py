"""Tabbed citation, retrieval, and grounding-trace panels."""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd

from autokg_rag.schemas.api import AnswerPayload


def render_citation_tabs(st: Any, payload: AnswerPayload) -> None:
    """Render citation, retrieval-hit, and grounding-trace tabs."""

    citations_tab, hits_tab, trace_tab = st.tabs(
        ["Citations", "Retrieval Hits", "Grounding Trace"]
    )

    with citations_tab:
        citation_rows = [
            {
                "chunk_id": citation.chunk_id,
                "doc_id": citation.doc_id,
                "page": citation.page,
                "section": citation.section,
            }
            for citation in payload.answer.citations
        ]
        citations_df = pd.DataFrame(citation_rows)
        st.dataframe(citations_df, width="stretch")

    with hits_tab:
        hit_rows = [
            {
                "rank": hit.rank,
                "score": round(hit.score, 4),
                "vector_score": round(hit.vector_score, 4),
                "graph_score": round(hit.graph_score, 4),
                "chunk_id": hit.chunk_id,
                "section": hit.section,
            }
            for hit in payload.hits
        ]
        hits_df = pd.DataFrame(hit_rows)
        st.dataframe(hits_df, width="stretch")

    with trace_tab:
        if not payload.citation_trace:
            st.caption("No grounding trace available.")
            return

        for trace in payload.citation_trace:
            support_score = float(trace.support_score)
            support_pct = int(max(0.0, min(1.0, support_score)) * 100)
            citation = trace.citation
            expander_title = (
                f"Sentence {trace.answer_sentence_id} · Support {support_score:.2f}"
            )

            with st.expander(expander_title):
                st.markdown(
                    (
                        '<div class="trace-meta">'
                        f"Citation: <code>{escape(citation.chunk_id)}</code> · "
                        f"Doc: <code>{escape(citation.doc_id)}</code> · "
                        f"Page: {citation.page} · "
                        f"Section: {escape(citation.section)}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    (
                        '<div class="support-bar" role="progressbar" '
                        f'aria-valuenow="{support_pct}" aria-valuemin="0" aria-valuemax="100">'
                        f'<span style="width: {support_pct}%;"></span>'
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
