"""Answer-card rendering for the redesigned Streamlit app."""

from __future__ import annotations

from html import escape
from typing import Any

from autokg_rag.schemas.api import AnswerPayload


def render_answer_card(st: Any, payload: AnswerPayload, *, elapsed_seconds: float | None) -> None:
    """Render the answer text with concise grounding metrics."""

    citation_count = len(payload.answer.citations)
    grounded_count = len(payload.citation_trace)
    answer_html = escape(payload.answer.answer_text).replace("\n", "<br>")

    timing_html = ""
    if elapsed_seconds is not None:
        timing_html = f'<div class="answer-timing">Answered in {elapsed_seconds:.2f}s</div>'

    with st.container():
        st.markdown(
            (
                '<section class="answer-card">'
                '<div class="answer-title">Answer</div>'
                f'<div class="answer-text">{answer_html}</div>'
                '<div class="metric-pill-row">'
                f'<span class="metric-pill">Citations: {citation_count}</span>'
                f'<span class="metric-pill">Grounded Sentences: {grounded_count}</span>'
                "</div>"
                f"{timing_html}"
                "</section>"
            ),
            unsafe_allow_html=True,
        )
