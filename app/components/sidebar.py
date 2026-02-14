"""Sidebar controls for the redesigned AutoRAG Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from autokg_rag.schemas.api import QueryMode


@dataclass(frozen=True)
class SidebarState:
    """UI state selected from the app sidebar."""

    run_id: str
    mode: QueryMode
    top_k: int


def _sidebar(st: Any) -> Any:
    return getattr(st, "sidebar", st)


def render_sidebar(st: Any) -> SidebarState:
    """Render sidebar controls and return selected query parameters."""

    sidebar = _sidebar(st)
    session_state = getattr(st, "session_state", {})
    available_runs = list(session_state.get("available_runs", []))
    previous_run = str(session_state.get("selected_run_id", ""))

    sidebar.markdown('<div class="sidebar-title">AutoRAG</div>', unsafe_allow_html=True)
    sidebar.markdown(
        '<div class="sidebar-subtitle">Milestone 6 Demo · v0.1</div>',
        unsafe_allow_html=True,
    )

    run_count = len(available_runs)
    run_label = f"Run ID ({run_count})"

    if available_runs:
        selected_index = run_count - 1
        if previous_run in available_runs:
            selected_index = available_runs.index(previous_run)
        selected_run = sidebar.selectbox(
            run_label,
            options=available_runs,
            index=selected_index,
        )
    else:
        sidebar.selectbox(
            run_label,
            options=["No runs available"],
            index=0,
            disabled=True,
        )
        selected_run = ""

    selected_mode = sidebar.radio(
        "Retrieval Mode",
        options=["vector", "graph", "hybrid"],
        index=2,
        horizontal=True,
    )
    top_k = int(sidebar.slider("Top-K", min_value=1, max_value=20, value=8, step=1))

    if available_runs:
        sidebar.markdown(
            (
                '<div class="health-badge health-ok">'
                '<span class="dot"></span>'
                "Artifacts detected"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    else:
        sidebar.markdown(
            (
                '<div class="health-badge health-warn">'
                '<span class="dot"></span>'
                "No artifacts found"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    session_state["selected_run_id"] = selected_run

    return SidebarState(
        run_id=selected_run,
        mode=cast(QueryMode, selected_mode),
        top_k=top_k,
    )
