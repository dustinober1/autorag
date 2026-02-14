"""Milestone 6 Streamlit app with a professional minimalist layout."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from pydantic import ValidationError

try:
    from app.components import (
        render_answer_card,
        render_arxiv_panel,
        render_citation_tabs,
        render_document_manager,
        render_model_selector,
        render_no_query_state,
        render_no_runs_state,
        render_question_bar,
        render_sidebar,
        render_store_manager,
        render_upload_panel,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path.
    from components import (  # type: ignore[no-redef]
        render_answer_card,
        render_arxiv_panel,
        render_citation_tabs,
        render_document_manager,
        render_model_selector,
        render_no_query_state,
        render_no_runs_state,
        render_question_bar,
        render_sidebar,
        render_store_manager,
        render_upload_panel,
    )
from autokg_rag.app_api.endpoints import (
    add_documents_endpoint,
    create_store_endpoint,
    delete_store_endpoint,
    ingest_arxiv_endpoint,
    list_documents_endpoint,
    list_ollama_models_endpoint,
    list_stores_endpoint,
    query_endpoint,
    remove_document_endpoint,
    search_arxiv_endpoint,
    upload_documents_endpoint,
)
from autokg_rag.config import Settings, load_settings
from autokg_rag.schemas.api import (
    AnswerPayload,
    ArxivPaper,
    OllamaModelInfo,
    QueryMode,
    StoreInfo,
)

_DEFAULT_QUESTION = "Compare mitigation and acceptance strategies."
_STYLE_FILES = ("main.css", "components.css")

_SESSION_AVAILABLE_RUNS = "available_runs"
_SESSION_LAST_RESULT = "latest_result"
_SESSION_LAST_ELAPSED = "latest_elapsed_seconds"
_SESSION_LAST_ERROR = "latest_error"
_SESSION_LAST_QUESTION = "latest_question"
_SESSION_LAST_RESULT_CONTEXT = "latest_result_context"
_SESSION_OLLAMA_MODELS = "ollama_models"
_SESSION_OLLAMA_HEALTH = "ollama_healthy"
_SESSION_SELECTED_ANSWER_MODEL = "selected_answer_model"
_SESSION_SELECTED_EMBEDDING_MODEL = "selected_embedding_model"
_SESSION_SELECTED_RERANKER_MODEL = "selected_reranker_model"
_SESSION_ARXIV_RESULTS = "arxiv_results"
_SESSION_ARXIV_SELECTED_LABELS = "arxiv_selected_labels"


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


def _notify_success(st: Any, message: str) -> None:
    if hasattr(st, "success"):
        st.success(message)
        return
    _render_info_card(st, message)


def _notify_error(st: Any, message: str) -> None:
    if hasattr(st, "error"):
        st.error(message)
        return
    _render_error_card(st, message)


def _restore_models(raw_models: Any) -> list[OllamaModelInfo]:
    if not isinstance(raw_models, list):
        return []
    models: list[OllamaModelInfo] = []
    for row in raw_models:
        if not isinstance(row, dict):
            continue
        try:
            models.append(OllamaModelInfo.model_validate(row))
        except ValidationError:
            continue
    return models


def _restore_arxiv_results(raw_papers: Any) -> list[ArxivPaper]:
    if not isinstance(raw_papers, list):
        return []
    papers: list[ArxivPaper] = []
    for row in raw_papers:
        if not isinstance(row, dict):
            continue
        try:
            papers.append(ArxivPaper.model_validate(row))
        except ValidationError:
            continue
    return papers


def _store_infos_from_run_ids(run_ids: Sequence[str]) -> list[StoreInfo]:
    return [
        StoreInfo(
            store_name=run_id,
            doc_count=0,
            chunk_count=0,
            has_embeddings=False,
            created_at=datetime(1970, 1, 1, tzinfo=UTC),
        )
        for run_id in run_ids
    ]


def _load_ollama_models(st: Any, *, settings: Settings) -> tuple[list[OllamaModelInfo], bool]:
    try:
        models = list_ollama_models_endpoint(settings=settings)
        healthy = True
    except Exception:
        models = []
        healthy = False

    st.session_state[_SESSION_OLLAMA_MODELS] = [model.model_dump(mode="json") for model in models]
    st.session_state[_SESSION_OLLAMA_HEALTH] = healthy
    return models, healthy


def _effective_settings_from_session(
    *,
    settings: Settings,
    session_state: dict[str, Any],
) -> Settings:
    answer_model = str(
        session_state.get(_SESSION_SELECTED_ANSWER_MODEL, settings.answer_model)
    ).strip()
    embedding_model = str(
        session_state.get(_SESSION_SELECTED_EMBEDDING_MODEL, settings.embedding_model)
    ).strip()
    reranker_model = str(
        session_state.get(_SESSION_SELECTED_RERANKER_MODEL, settings.reranker_model)
    ).strip()
    return settings.model_copy(
        update={
            "answer_model": answer_model or settings.answer_model,
            "embedding_model": embedding_model or settings.embedding_model,
            "reranker_model": reranker_model or settings.reranker_model,
        }
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

    management_enabled = run_ids is None
    if management_enabled:
        stores = list_stores_endpoint(settings=resolved_settings)
    else:
        stores = _store_infos_from_run_ids(list(run_ids or []))

    if management_enabled:
        manager_state = render_store_manager(
            st,
            stores=stores,
            default_store=str(session_state.get("selected_run_id", "")),
        )
        if manager_state.create_requested and manager_state.create_name:
            try:
                create_store_endpoint(
                    store_name=manager_state.create_name,
                    settings=resolved_settings,
                )
                _notify_success(st, f"Created store '{manager_state.create_name}'.")
            except Exception as exc:  # noqa: BLE001
                _notify_error(st, f"Create store failed: {exc}")
            stores = list_stores_endpoint(settings=resolved_settings)

        if manager_state.delete_requested and manager_state.selected_store:
            try:
                deleted = delete_store_endpoint(
                    store_name=manager_state.selected_store,
                    settings=resolved_settings,
                )
                if deleted:
                    _notify_success(st, f"Deleted store '{manager_state.selected_store}'.")
                else:
                    _notify_error(st, f"Store not found: {manager_state.selected_store}")
            except Exception as exc:  # noqa: BLE001
                _notify_error(st, f"Delete store failed: {exc}")
            stores = list_stores_endpoint(settings=resolved_settings)

        cached_models = _restore_models(session_state.get(_SESSION_OLLAMA_MODELS))
        cached_health = bool(session_state.get(_SESSION_OLLAMA_HEALTH, False))
        if not cached_models and _SESSION_OLLAMA_MODELS not in session_state:
            cached_models, cached_health = _load_ollama_models(st, settings=resolved_settings)

        model_state = render_model_selector(
            st,
            models=cached_models,
            ollama_healthy=cached_health,
            default_answer_model=str(
                session_state.get(_SESSION_SELECTED_ANSWER_MODEL, resolved_settings.answer_model)
            ),
            default_embedding_model=str(
                session_state.get(
                    _SESSION_SELECTED_EMBEDDING_MODEL,
                    resolved_settings.embedding_model,
                )
            ),
            default_reranker_model=str(
                session_state.get(
                    _SESSION_SELECTED_RERANKER_MODEL,
                    resolved_settings.reranker_model,
                )
            ),
        )
        if model_state.refresh_requested:
            cached_models, cached_health = _load_ollama_models(st, settings=resolved_settings)
            model_state = model_state.__class__(
                answer_model=model_state.answer_model,
                embedding_model=model_state.embedding_model,
                reranker_model=model_state.reranker_model,
                refresh_requested=False,
            )

        session_state[_SESSION_SELECTED_ANSWER_MODEL] = model_state.answer_model
        session_state[_SESSION_SELECTED_EMBEDDING_MODEL] = model_state.embedding_model
        session_state[_SESSION_SELECTED_RERANKER_MODEL] = model_state.reranker_model

    available_runs = [store.store_name for store in stores]
    session_state[_SESSION_AVAILABLE_RUNS] = available_runs

    sidebar_state = render_sidebar(st)
    active_settings = _effective_settings_from_session(
        settings=resolved_settings,
        session_state=session_state,
    )

    run_token = escape(sidebar_state.run_id) if sidebar_state.run_id else "No run selected"
    st.markdown(
        (
            '<div class="context-line">'
            f'Run: <span class="context-token">{run_token}</span> '
            f'· Mode: <span class="context-token">{sidebar_state.mode}</span> '
            f'· Top-K: <span class="context-token">{sidebar_state.top_k}</span> '
            f'· Answer: <span class="context-token">{escape(active_settings.answer_model)}</span> '
            f'· Embed: <span class="context-token">{escape(active_settings.embedding_model)}</span>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if not available_runs:
        render_no_runs_state(st)
        return

    if management_enabled and sidebar_state.run_id:
        upload_state = render_upload_panel(st)
        if upload_state.upload_requested:
            try:
                with st.spinner("Uploading and indexing documents..."):
                    ingest_result = upload_documents_endpoint(
                        store_name=sidebar_state.run_id,
                        files=upload_state.files,
                        settings=active_settings,
                    )
                _notify_success(
                    st,
                    (
                        f"Uploaded {ingest_result.documents} documents "
                        f"({ingest_result.pages} pages, {ingest_result.chunks} chunks)."
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                _notify_error(st, f"Upload failed: {exc}")

        documents = list_documents_endpoint(
            store_name=sidebar_state.run_id,
            settings=active_settings,
        )
        document_state = render_document_manager(st, documents=documents)
        if document_state.remove_requested and document_state.remove_doc_id:
            try:
                removed = remove_document_endpoint(
                    store_name=sidebar_state.run_id,
                    doc_id=document_state.remove_doc_id,
                    settings=active_settings,
                )
                if removed:
                    _notify_success(st, f"Removed document '{document_state.remove_doc_id}'.")
                else:
                    _notify_error(st, f"Document not found: {document_state.remove_doc_id}")
            except Exception as exc:  # noqa: BLE001
                _notify_error(st, f"Remove failed: {exc}")

        if document_state.add_requested:
            try:
                with st.spinner("Adding documents..."):
                    add_result = add_documents_endpoint(
                        store_name=sidebar_state.run_id,
                        files=document_state.add_files,
                        settings=active_settings,
                    )
                _notify_success(
                    st,
                    (
                        f"Added {add_result.documents} documents "
                        f"({add_result.pages} pages, {add_result.chunks} chunks)."
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                _notify_error(st, f"Add documents failed: {exc}")

        arxiv_results = _restore_arxiv_results(session_state.get(_SESSION_ARXIV_RESULTS))
        arxiv_state = render_arxiv_panel(
            st,
            papers=arxiv_results,
            store_names=available_runs,
            default_store=sidebar_state.run_id,
        )
        if arxiv_state.search_requested and arxiv_state.query:
            try:
                with st.spinner("Searching arXiv..."):
                    arxiv_results = search_arxiv_endpoint(
                        query=arxiv_state.query,
                        max_results=arxiv_state.max_results,
                    )
                session_state[_SESSION_ARXIV_RESULTS] = [
                    paper.model_dump(mode="json") for paper in arxiv_results
                ]
                session_state[_SESSION_ARXIV_SELECTED_LABELS] = []
                _notify_success(st, f"Found {len(arxiv_results)} arXiv papers.")
            except Exception as exc:  # noqa: BLE001
                _notify_error(st, f"arXiv search failed: {exc}")

        if arxiv_state.import_requested and arxiv_state.selected_ids:
            if not arxiv_state.target_store:
                _notify_error(st, "Select a target store for arXiv import.")
            else:
                selected_map = {paper.arxiv_id: paper for paper in arxiv_results}
                selected_papers = [
                    selected_map[paper_id]
                    for paper_id in arxiv_state.selected_ids
                    if paper_id in selected_map
                ]
                if selected_papers:
                    try:
                        with st.spinner("Downloading and ingesting selected arXiv papers..."):
                            import_result = ingest_arxiv_endpoint(
                                store_name=arxiv_state.target_store,
                                papers=selected_papers,
                                settings=active_settings,
                            )
                        _notify_success(
                            st,
                            (
                                f"Imported {import_result.documents} arXiv papers "
                                f"({import_result.pages} pages, {import_result.chunks} chunks) "
                                f"into '{arxiv_state.target_store}'."
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001
                        _notify_error(st, f"arXiv import failed: {exc}")
                else:
                    _notify_error(st, "No valid arXiv papers were selected for import.")

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
                        settings=active_settings,
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
