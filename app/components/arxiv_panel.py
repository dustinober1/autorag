"""arXiv search/select/import panel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from autokg_rag.schemas.api import ArxivPaper


@dataclass(frozen=True)
class ArxivPanelState:
    """arXiv panel state emitted from UI controls."""

    query: str
    max_results: int
    target_store: str
    search_requested: bool
    selected_ids: list[str]
    import_requested: bool


_SELECTED_LABELS_KEY = "arxiv_selected_labels"


def render_arxiv_panel(
    st: Any,
    *,
    papers: Sequence[ArxivPaper],
    store_names: Sequence[str],
    default_store: str,
) -> ArxivPanelState:
    """Render arXiv search controls and selected-paper import actions."""

    st.markdown("### arXiv Search & Import")
    stores = list(store_names)
    if stores:
        default_index = stores.index(default_store) if default_store in stores else len(stores) - 1
        target_store = str(
            st.selectbox(
                "Target Store for Import",
                options=stores,
                index=max(0, default_index),
            )
        )
    else:
        st.selectbox(
            "Target Store for Import",
            options=["No stores available"],
            index=0,
            disabled=True,
        )
        target_store = ""

    query = st.text_input(
        "Search arXiv",
        value="",
        placeholder="e.g., retrieval augmented generation",
    )
    max_results = int(st.slider("Max Results", min_value=1, max_value=50, value=10, step=1))
    search_requested = st.button("Search arXiv", use_container_width=True)

    selected_ids: list[str] = []
    if papers:
        result_rows = [
            {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": ", ".join(paper.authors[:3]),
                "published": paper.published.date().isoformat(),
            }
            for paper in papers
        ]
        st.dataframe(pd.DataFrame(result_rows), width="stretch")

        option_labels = [f"{paper.arxiv_id} · {paper.title[:80]}" for paper in papers]
        label_to_id = {
            label: paper.arxiv_id
            for label, paper in zip(option_labels, papers, strict=False)
        }

        action_col_left, action_col_right = st.columns([1, 1])
        with action_col_left:
            select_all_requested = bool(st.button("Select All Results", use_container_width=True))
        with action_col_right:
            clear_selection_requested = bool(st.button("Clear Selection", use_container_width=True))

        if select_all_requested:
            st.session_state[_SELECTED_LABELS_KEY] = list(option_labels)
        elif clear_selection_requested:
            st.session_state[_SELECTED_LABELS_KEY] = []

        selected_labels = list(
            st.multiselect(
                "Select papers to import",
                options=option_labels,
                default=[],
                key=_SELECTED_LABELS_KEY,
            )
        )
        selected_ids = [label_to_id[label] for label in selected_labels if label in label_to_id]
    else:
        st.caption("No search results yet.")

    import_requested = st.button(
        "Import Selected Papers",
        use_container_width=True,
        disabled=not bool(selected_ids),
    )

    return ArxivPanelState(
        query=query.strip(),
        max_results=max_results,
        target_store=target_store,
        search_requested=bool(search_requested),
        selected_ids=selected_ids,
        import_requested=bool(import_requested),
    )
