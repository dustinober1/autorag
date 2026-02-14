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
    search_requested: bool
    selected_ids: list[str]
    import_requested: bool


def render_arxiv_panel(st: Any, *, papers: Sequence[ArxivPaper]) -> ArxivPanelState:
    """Render arXiv search controls and selected-paper import actions."""

    st.markdown("### arXiv Search & Import")
    query = st.text_input(
        "Search arXiv",
        value="",
        placeholder="e.g., retrieval augmented generation",
    )
    max_results = int(st.slider("Max Results", min_value=1, max_value=50, value=10, step=1))
    search_requested = st.button("Search arXiv", use_container_width=True)

    selected_ids: list[str] = []
    if papers:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "authors": ", ".join(paper.authors[:3]),
                        "published": paper.published.date().isoformat(),
                    }
                    for paper in papers
                ]
            ),
            width="stretch",
        )
        selected_ids = list(
            st.multiselect(
                "Select papers to import",
                options=[paper.arxiv_id for paper in papers],
                default=[],
            )
        )
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
        search_requested=bool(search_requested),
        selected_ids=selected_ids,
        import_requested=bool(import_requested),
    )

