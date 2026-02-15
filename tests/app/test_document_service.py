from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pytest import MonkeyPatch

from autokg_rag.app_api.document_service import add_documents, list_documents, remove_document
from autokg_rag.app_api.store_service import create_store
from autokg_rag.config import Settings
from autokg_rag.io import read_parquet_rows


@dataclass
class _Upload:
    name: str
    payload: bytes

    def getvalue(self) -> bytes:
        return self.payload


def _make_upload(name: str, body: str) -> _Upload:
    return _Upload(name=name, payload=body.encode("utf-8"))


def test_document_service_add_list_remove_and_keep_embeddings_aligned(tmp_path: Path) -> None:
    settings = Settings(artifact_root=tmp_path / "artifacts")
    create_store("demo", settings)

    upload_a = _make_upload(
        "a.pdf",
        "Scope control aligns with objectives.\fRisk mitigation reduces impact.",
    )
    upload_b = _make_upload(
        "b.pdf",
        "Acceptance strategy tolerates residual risk.",
    )

    add_result = add_documents("demo", [upload_a, upload_b], settings)
    assert add_result.documents == 2
    assert add_result.pages >= 2
    assert add_result.chunks >= 2

    docs = list_documents("demo", settings)
    assert len(docs) == 2
    assert all(doc.page_count >= 1 for doc in docs)
    assert all(doc.chunk_count >= 1 for doc in docs)
    assert all(doc.document_type == "generic" for doc in docs)

    duplicate_result = add_documents("demo", [upload_a], settings)
    assert duplicate_result.documents == 0
    assert duplicate_result.skipped_duplicates == 1

    removed = remove_document("demo", docs[0].doc_id, settings)
    assert removed is True

    docs_after_remove = list_documents("demo", settings)
    assert len(docs_after_remove) == 1

    artifact_dir = settings.artifact_root / "demo"
    chunk_rows = read_parquet_rows(artifact_dir / "chunks.parquet")
    meta_rows = read_parquet_rows(artifact_dir / "embedding_meta.parquet")
    matrix = np.load(artifact_dir / "embeddings.npy")

    assert len(meta_rows) == len(chunk_rows)
    assert int(matrix.shape[0]) == len(meta_rows)


def test_document_service_detects_pmbok_document_type_from_filename(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = Settings(artifact_root=tmp_path / "artifacts")
    create_store("demo", settings)
    monkeypatch.setattr(
        "autokg_rag.app_api.document_service.initialize_pmbok_toc_for_document",
        lambda _path: None,
    )

    pmbok_upload = _make_upload(
        "pmbok_scope.pdf",
        "Scope control guidance.\fRisk response guidance.",
    )
    add_result = add_documents("demo", [pmbok_upload], settings)
    assert add_result.documents == 1

    docs = list_documents("demo", settings)
    assert len(docs) == 1
    assert docs[0].document_type == "pmbok"
