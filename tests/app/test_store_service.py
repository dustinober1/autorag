from __future__ import annotations

from pathlib import Path

import numpy as np

from autokg_rag.app_api.store_service import (
    create_store,
    delete_store,
    get_store_info,
    list_stores,
)
from autokg_rag.config import Settings
from autokg_rag.io import write_parquet_rows


def test_store_service_create_list_get_delete(tmp_path: Path) -> None:
    settings = Settings(artifact_root=tmp_path / "artifacts")

    store_name = create_store("demo_store", settings)
    assert store_name == "demo_store"

    store_dir = settings.artifact_root / store_name
    assert store_dir.exists()

    write_parquet_rows(
        store_dir / "documents.parquet",
        [
            {
                "doc_id": "doc_1",
                "title": "Doc One",
                "source_path": "doc_1.pdf",
                "sha256": "a" * 64,
            }
        ],
    )
    write_parquet_rows(
        store_dir / "chunks.parquet",
        [
            {
                "chunk_id": "doc_1-p1-heading_recursive-c1",
                "doc_id": "doc_1",
                "page": 1,
                "section": "section",
                "chunk_text": "chunk text",
            }
        ],
    )
    write_parquet_rows(
        store_dir / "embedding_meta.parquet",
        [
            {
                "chunk_id": "doc_1-p1-heading_recursive-c1",
                "row_idx": 0,
                "provider": "local_hash",
                "embedding_model": "bge-small-en-v1.5",
                "dim": 16,
            }
        ],
    )
    np.save(store_dir / "embeddings.npy", np.zeros((1, 16), dtype=np.float32))

    info = get_store_info("demo_store", settings)
    assert info.store_name == "demo_store"
    assert info.doc_count == 1
    assert info.chunk_count == 1
    assert info.has_embeddings is True

    stores = list_stores(settings)
    assert [store.store_name for store in stores] == ["demo_store"]

    assert delete_store("demo_store", settings) is True
    assert delete_store("demo_store", settings) is False

