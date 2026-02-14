from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autokg_rag.app_api.endpoints import upload_documents_endpoint
from autokg_rag.app_api.service import query_service
from autokg_rag.config import Settings
from autokg_rag.schemas.api import QueryRequest


@dataclass
class _Upload:
    name: str
    payload: bytes

    def getvalue(self) -> bytes:
        return self.payload


def test_upload_to_query_flow_returns_grounded_payload(tmp_path: Path) -> None:
    settings = Settings(artifact_root=tmp_path / "artifacts")
    upload = _Upload(
        name="fixture.pdf",
        payload=(
            b"Mitigation reduces probability and impact.\f"
            b"Acceptance keeps a contingency plan for residual risk."
        ),
    )

    ingest_result = upload_documents_endpoint(
        store_name="ui_store",
        files=[upload],
        settings=settings,
    )
    assert ingest_result.documents == 1
    assert ingest_result.chunks >= 1

    payload = query_service(
        request=QueryRequest(
            run_id="ui_store",
            question="How do mitigation and acceptance differ?",
            mode="vector",
            top_k=3,
        ),
        settings=settings,
    )
    assert payload.answer.answer_text.strip()
    assert payload.answer.citations
    assert payload.hits
