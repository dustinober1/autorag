"""Per-document list/add/remove management panel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from autokg_rag.schemas.api import DocumentInfo


@dataclass(frozen=True)
class DocumentManagerState:
    """Document management action state."""

    add_files: list[Any]
    add_requested: bool
    remove_doc_id: str
    remove_requested: bool
    refresh_requested: bool


def render_document_manager(st: Any, *, documents: Sequence[DocumentInfo]) -> DocumentManagerState:
    """Render document list with add/remove actions."""

    st.markdown("### Document Manager")
    if documents:
        rows = [
            {
                "title": doc.title,
                "doc_id": doc.doc_id,
                "pages": doc.page_count,
                "chunks": doc.chunk_count,
            }
            for doc in documents
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.caption("No documents in this store yet.")

    doc_options = [doc.doc_id for doc in documents]
    if doc_options:
        remove_doc_id = str(st.selectbox("Document to Remove", options=doc_options, index=0))
    else:
        st.selectbox("Document to Remove", options=["No documents"], index=0, disabled=True)
        remove_doc_id = ""

    remove_requested = st.button(
        "Remove Selected Document",
        use_container_width=True,
        disabled=not bool(remove_doc_id),
    )
    add_files = list(
        st.file_uploader(
            "Add PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
        )
        or []
    )
    add_requested = st.button("Add Documents to Store", use_container_width=True)
    refresh_requested = st.button("Refresh Document List", use_container_width=True)

    return DocumentManagerState(
        add_files=add_files,
        add_requested=bool(add_requested),
        remove_doc_id=remove_doc_id,
        remove_requested=bool(remove_requested),
        refresh_requested=bool(refresh_requested),
    )

