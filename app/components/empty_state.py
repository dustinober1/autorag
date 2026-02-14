"""Empty-state renderers for onboarding and pre-query states."""

from __future__ import annotations

from typing import Any


def render_no_runs_state(st: Any) -> None:
    """Render onboarding guidance when no run artifacts are available."""

    st.markdown(
        (
            '<section class="empty-state">'
            '<div class="empty-title">No run artifacts available</div>'
            '<div class="empty-body">'
            "Run <code>make demo-build</code> to generate an interactive run, "
            "then refresh this page to start querying."
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_no_query_state(st: Any) -> None:
    """Render prompt text before the first question is submitted."""

    st.markdown(
        (
            '<section class="empty-state">'
            '<div class="empty-title">Ask a question to begin</div>'
            '<div class="empty-body">'
            "Use the question bar above to run retrieval and inspect citations, "
            "hits, and sentence-level grounding."
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )
