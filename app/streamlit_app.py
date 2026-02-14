"""Milestone 6 Streamlit app with a professional minimalist layout."""

from __future__ import annotations

from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence, cast

from pydantic import ValidationError

try:
    from app.components import (
        render_answer_card,
        render_citation_tabs,
        render_no_query_state,
        render_no_runs_state,
        render_question_bar,
        render_sidebar,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path.
    from components import (  # type: ignore[no-redef]
        render_answer_card,
        render_citation_tabs,
        render_no_query_state,
        render_no_runs_state,
        render_question_bar,
        render_sidebar,
    )
from autokg_rag.app_api.endpoints import list_runs_endpoint, query_endpoint
from autokg_rag.config import Settings, load_settings
from autokg_rag.schemas.api import AnswerPayload, QueryMode

_DEFAULT_QUESTION = "Compare mitigation and acceptance strategies."
_STYLE_FILES = ("main.css", "components.css")

_SESSION_AVAILABLE_RUNS = "available_runs"
_SESSION_LAST_RESULT = "latest_result"
_SESSION_LAST_ELAPSED = "latest_elapsed_seconds"
_SESSION_LAST_ERROR = "latest_error"
_SESSION_LAST_QUESTION = "latest_question"
_SESSION_LAST_RESULT_CONTEXT = "latest_result_context"


QueryHandler = Callable[..., AnswerPayload]


def _apply_styles(st: Any, *, style_dir: Path) -> None:
    for filename in _STYLE_FILES:
        css_path = style_dir / filename
        if not css_path.exists():
            continue

        css_text = css_path.read_text(encoding="utf-8").strip()
        if not css_text:
            continue

        st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)


def _restore_payload(raw_payload: Any) -> AnswerPayload | None:
    if isinstance(raw_payload, AnswerPayload):
        return raw_payload
    if isinstance(raw_payload, dict):
        try:
            return AnswerPayload.model_validate(raw_payload)
        except ValidationError:
            return None
    return None


def _restore_elapsed(raw_elapsed: Any) -> float | None:
    try:
        elapsed = float(raw_elapsed)
    except (TypeError, ValueError):
        return None
    if elapsed < 0:
        return None
    return elapsed


def _render_error_card(st: Any, message: str) -> None:
    body = escape(message)
    st.markdown(
        (
            '<section class="error-card">'
            '<div class="error-title">Query failed</div>'
            f'<div class="error-body">{body}</div>'
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def _restore_result_context(raw_context: Any) -> dict[str, Any]:
    if not isinstance(raw_context, dict):
        return {}
    return raw_context


def _render_info_card(st: Any, message: str) -> None:
    body = escape(message)
    st.markdown(
        (
            '<section class="info-card">'
            f"<div>{body}</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def _context_matches_sidebar(
    result_context: dict[str, Any],
    *,
    run_id: str,
    mode: str,
    top_k: int,
    question: str,
) -> bool:
    return (
        result_context.get("run_id") == run_id
        and result_context.get("mode") == mode
        and result_context.get("top_k") == top_k
        and result_context.get("question") == question
    )


def render_app(
    st: Any,
    *,
    settings: Settings | None = None,
    run_ids: Sequence[str] | None = None,
    query_handler: QueryHandler | None = None,
) -> None:
    """Render the redesigned Streamlit app with injectable dependencies for tests."""

    resolved_settings = load_settings() if settings is None else settings
    session_state = st.session_state

    style_dir = Path(__file__).with_name("styles")
    st.set_page_config(page_title="AutoRAG", layout="wide", initial_sidebar_state="expanded")
    _apply_styles(st, style_dir=style_dir)

    st.markdown('<div class="main-title">AutoRAG</div>', unsafe_allow_html=True)
    st.markdown(
        (
            '<div class="main-subtitle">'
            "Local-first retrieval with grounded answers and citation inspection."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    available_runs = (
        list(run_ids) if run_ids is not None else list_runs_endpoint(settings=resolved_settings)
    )
    session_state[_SESSION_AVAILABLE_RUNS] = available_runs

    sidebar_state = render_sidebar(st)

    run_token = escape(sidebar_state.run_id) if sidebar_state.run_id else "No run selected"
    st.markdown(
        (
            '<div class="context-line">'
            f'Run: <span class="context-token">{run_token}</span> '
            f'· Mode: <span class="context-token">{sidebar_state.mode}</span> '
            f'· Top-K: <span class="context-token">{sidebar_state.top_k}</span>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if not available_runs:
        render_no_runs_state(st)
        return

    default_question = str(session_state.get(_SESSION_LAST_QUESTION, _DEFAULT_QUESTION))
    question, submitted = render_question_bar(st, default_question=default_question)

    if submitted:
        if not question:
            session_state[_SESSION_LAST_ERROR] = "Question cannot be empty."
        else:
            session_state[_SESSION_LAST_QUESTION] = question
            active_handler = query_endpoint if query_handler is None else query_handler
            try:
                with st.spinner("Running retrieval and grounding..."):
                    started_at = perf_counter()
                    payload = active_handler(
                        run_id=sidebar_state.run_id,
                        question=question,
                        mode=cast(QueryMode, sidebar_state.mode),
                        top_k=sidebar_state.top_k,
                        settings=resolved_settings,
                    )
                    elapsed_seconds = perf_counter() - started_at
            except Exception as exc:  # noqa: BLE001
                session_state[_SESSION_LAST_ERROR] = str(exc)
            else:
                session_state[_SESSION_LAST_ERROR] = ""
                session_state[_SESSION_LAST_RESULT] = payload.model_dump(mode="json")
                session_state[_SESSION_LAST_ELAPSED] = elapsed_seconds
                session_state[_SESSION_LAST_RESULT_CONTEXT] = {
                    "run_id": sidebar_state.run_id,
                    "mode": sidebar_state.mode,
                    "top_k": sidebar_state.top_k,
                    "question": question,
                }

    error_message = str(session_state.get(_SESSION_LAST_ERROR, "")).strip()
    if error_message:
        _render_error_card(st, error_message)

    cached_payload = _restore_payload(session_state.get(_SESSION_LAST_RESULT))
    if cached_payload is None:
        render_no_query_state(st)
        return

    result_context = _restore_result_context(session_state.get(_SESSION_LAST_RESULT_CONTEXT))
    if result_context:
        if not _context_matches_sidebar(
            result_context,
            run_id=sidebar_state.run_id,
            mode=sidebar_state.mode,
            top_k=sidebar_state.top_k,
            question=question,
        ):
            _render_info_card(
                st,
                (
                    "Showing latest result from "
                    f"Run: {result_context.get('run_id', 'unknown')} · "
                    f"Mode: {result_context.get('mode', 'unknown')} · "
                    f"Top-K: {result_context.get('top_k', 'unknown')}. "
                    "Submit again to refresh with current controls."
                ),
            )
        elif error_message:
            _render_info_card(st, "Showing the latest successful result below.")

    elapsed_seconds = _restore_elapsed(session_state.get(_SESSION_LAST_ELAPSED))
    render_answer_card(st, cached_payload, elapsed_seconds=elapsed_seconds)
    render_citation_tabs(st, cached_payload)


def main() -> None:
    """Execute Streamlit app entrypoint."""

    import streamlit as st

    render_app(st)


if __name__ == "__main__":
    main()
