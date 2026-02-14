"""Milestone 6 Streamlit demo app."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, cast

from app.components import StreamlitLike, apply_styles, render_answer_panel, render_citation_panel
from autokg_rag.app_api.endpoints import list_runs_endpoint, query_endpoint
from autokg_rag.config import Settings, load_settings
from autokg_rag.schemas.api import AnswerPayload, QueryMode

_DEFAULT_QUESTION = "Compare mitigation and acceptance strategies."


class _StreamlitShell(StreamlitLike, Protocol):
    """Interface needed by this module for rendering and controls."""

    def set_page_config(self, **kwargs: object) -> Any: ...

    def title(self, body: str) -> Any: ...

    def caption(self, body: str) -> Any: ...

    def warning(self, body: str) -> Any: ...

    def error(self, body: str) -> Any: ...

    def info(self, body: str) -> Any: ...

    def selectbox(self, label: str, options: Sequence[str], index: int = 0) -> str: ...

    def radio(
        self,
        label: str,
        options: Sequence[str],
        index: int = 0,
        horizontal: bool = False,
    ) -> str: ...

    def text_input(self, label: str, value: str = "") -> str: ...

    def slider(
        self,
        label: str,
        min_value: int,
        max_value: int,
        value: int,
        step: int = 1,
    ) -> int: ...

    def button(self, label: str) -> bool: ...


QueryHandler = Callable[..., AnswerPayload]


def render_app(
    st: _StreamlitShell,
    *,
    settings: Settings | None = None,
    run_ids: Sequence[str] | None = None,
    query_handler: QueryHandler | None = None,
) -> None:
    """Render the Streamlit app with injectable dependencies for tests."""

    resolved_settings = load_settings() if settings is None else settings
    style_path = Path(__file__).with_name("styles.css")

    st.set_page_config(page_title="AutoRAG M6 Demo", layout="wide")
    apply_styles(st, css_path=style_path)
    st.title("AutoRAG Milestone 6 Demo")
    st.caption("Local-first retrieval demo with answer grounding and citation inspection.")

    available_runs = (
        list(run_ids)
        if run_ids is not None
        else list_runs_endpoint(settings=resolved_settings)
    )

    st.subheader("Query Panel")
    if not available_runs:
        st.warning(
            "No run artifacts found. Run `make demo-build` first to generate an interactive run."
        )
        st.subheader("Citation Viewer")
        st.info("Citations will appear here after a query is executed.")
        return

    selected_run = st.selectbox(
        "Run ID",
        options=available_runs,
        index=max(0, len(available_runs) - 1),
    )
    selected_mode = st.radio(
        "Retrieval Mode",
        options=["vector", "graph", "hybrid"],
        index=2,
        horizontal=True,
    )
    question = st.text_input("Question", value=_DEFAULT_QUESTION)
    top_k = int(st.slider("Top-K", min_value=1, max_value=20, value=8, step=1))

    payload: AnswerPayload | None = None
    if st.button("Run Query"):
        active_handler = query_endpoint if query_handler is None else query_handler
        try:
            payload = active_handler(
                run_id=selected_run,
                question=question,
                mode=cast(QueryMode, selected_mode),
                top_k=top_k,
                settings=resolved_settings,
            )
        except Exception as exc:  # noqa: BLE001 - Streamlit should surface all request errors.
            st.error(str(exc))

    if payload is None:
        st.subheader("Citation Viewer")
        st.info("Submit a question to render answer and citation panels.")
        return

    render_answer_panel(st, payload)
    st.subheader("Citation Viewer")
    render_citation_panel(st, payload)


def main() -> None:
    """Execute Streamlit app entrypoint."""

    import streamlit as st

    render_app(st)


if __name__ == "__main__":
    main()
