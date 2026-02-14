"""Sidebar store CRUD controls."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from autokg_rag.schemas.api import StoreInfo


@dataclass(frozen=True)
class StoreManagerState:
    """Store manager control state emitted from sidebar."""

    selected_store: str
    create_name: str
    create_requested: bool
    delete_requested: bool


def _sidebar(st: Any) -> Any:
    return getattr(st, "sidebar", st)


def render_store_manager(
    st: Any,
    *,
    stores: Sequence[StoreInfo],
    default_store: str = "",
) -> StoreManagerState:
    """Render store CRUD controls and return requested actions."""

    sidebar = _sidebar(st)
    options = [store.store_name for store in stores]
    selected_store = default_store if default_store in options else (options[-1] if options else "")

    sidebar.markdown("### Store Manager")
    if options:
        selected_store = sidebar.selectbox(
            "Target Store",
            options=options,
            index=options.index(selected_store),
        )
    else:
        sidebar.selectbox("Target Store", options=["No stores yet"], index=0, disabled=True)
        selected_store = ""

    create_name = sidebar.text_input("New Store Name", value="")
    create_requested = sidebar.button("Create Store", use_container_width=True)
    delete_requested = sidebar.button(
        "Delete Selected Store",
        use_container_width=True,
        disabled=not bool(selected_store),
    )

    if selected_store:
        info = next((store for store in stores if store.store_name == selected_store), None)
        if info is not None:
            embeddings_text = "Yes" if info.has_embeddings else "No"
            sidebar.markdown(
                (
                    '<div class="store-stats">'
                    f"<div>Docs: <strong>{info.doc_count}</strong></div>"
                    f"<div>Chunks: <strong>{info.chunk_count}</strong></div>"
                    f"<div>Embeddings: <strong>{embeddings_text}</strong></div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    return StoreManagerState(
        selected_store=selected_store,
        create_name=create_name.strip(),
        create_requested=bool(create_requested),
        delete_requested=bool(delete_requested),
    )
