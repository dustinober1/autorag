"""Question input and submit controls for the main app area."""

from __future__ import annotations

from typing import Any


def render_question_bar(st: Any, *, default_question: str) -> tuple[str, bool]:
    """Render a full-width question input with submit button."""

    st.markdown('<div class="question-label">Question</div>', unsafe_allow_html=True)
    input_col, button_col = st.columns([7, 1])

    with input_col:
        question = st.text_input(
            "Question",
            value=default_question,
            placeholder="Ask a grounded question about your corpus.",
            label_visibility="collapsed",
        )

    with button_col:
        submitted = st.button("Submit", use_container_width=True)

    return question.strip(), submitted
