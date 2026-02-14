"""Main-area multi-file upload panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UploadPanelState:
    """Upload action state."""

    files: list[Any]
    upload_requested: bool


def render_upload_panel(st: Any) -> UploadPanelState:
    """Render upload controls and return selected files + action state."""

    st.markdown("### Upload Documents")
    files = list(
        st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
        )
        or []
    )
    upload_requested = st.button("Ingest Uploaded Files", use_container_width=True)
    return UploadPanelState(files=files, upload_requested=bool(upload_requested))

