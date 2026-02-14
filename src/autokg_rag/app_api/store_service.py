"""Store CRUD and metadata helpers for Streamlit app management."""

from __future__ import annotations

import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from autokg_rag.config import Settings
from autokg_rag.exceptions import IngestError
from autokg_rag.io import read_parquet_rows, write_parquet_rows
from autokg_rag.schemas.api import StoreInfo

_STORE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_META_FILE = "store_meta.json"


def _validate_store_name(store_name: str) -> str:
    normalized = store_name.strip()
    if not normalized:
        raise IngestError("Store name must not be empty.")
    if not _STORE_NAME_RE.fullmatch(normalized):
        raise IngestError(
            "Invalid store name. Use 1-64 characters: letters, numbers, '_', '-', '.'."
        )
    return normalized


def _store_dir(*, store_name: str, settings: Settings) -> Path:
    return settings.artifact_root / _validate_store_name(store_name)


def _created_at_for_dir(store_dir: Path) -> datetime:
    meta_path = store_dir / _META_FILE
    if meta_path.exists():
        try:
            raw = meta_path.read_text(encoding="utf-8").strip()
            parsed = datetime.fromisoformat(raw)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        except (OSError, ValueError):
            pass

    stat = store_dir.stat()
    return datetime.fromtimestamp(stat.st_mtime, tz=UTC)


def _has_embeddings(store_dir: Path) -> bool:
    embeddings_path = store_dir / "embeddings.npy"
    meta_path = store_dir / "embedding_meta.parquet"
    if not embeddings_path.exists() or not meta_path.exists():
        return False

    try:
        meta_rows = read_parquet_rows(meta_path)
        matrix = np.load(embeddings_path)
    except OSError:
        return False

    if matrix.ndim != 2:
        return False
    return int(matrix.shape[0]) == len(meta_rows)


def _to_store_info(store_dir: Path) -> StoreInfo:
    document_rows = read_parquet_rows(store_dir / "documents.parquet")
    chunk_rows = read_parquet_rows(store_dir / "chunks.parquet")
    return StoreInfo(
        store_name=store_dir.name,
        doc_count=len(document_rows),
        chunk_count=len(chunk_rows),
        has_embeddings=_has_embeddings(store_dir),
        created_at=_created_at_for_dir(store_dir),
    )


def create_store(store_name: str, settings: Settings) -> str:
    """Create an empty store directory with baseline artifact files."""

    normalized_name = _validate_store_name(store_name)
    store_dir = _store_dir(store_name=normalized_name, settings=settings)
    if store_dir.exists():
        raise IngestError(f"Store already exists: {normalized_name}")

    store_dir.mkdir(parents=True, exist_ok=False)
    created_at = datetime.now(tz=UTC).isoformat()
    (store_dir / _META_FILE).write_text(created_at, encoding="utf-8")

    # Baseline empty artifacts ensure list/count operations work before first ingest.
    write_parquet_rows(store_dir / "documents.parquet", [])
    write_parquet_rows(store_dir / "pages.parquet", [])
    write_parquet_rows(store_dir / "chunks.parquet", [])

    return normalized_name


def delete_store(store_name: str, settings: Settings) -> bool:
    """Delete one store directory and all contained artifacts."""

    store_dir = _store_dir(store_name=store_name, settings=settings)
    if not store_dir.exists():
        return False
    if not store_dir.is_dir():
        raise IngestError(f"Store path is not a directory: {store_dir}")

    shutil.rmtree(store_dir)
    return True


def list_stores(settings: Settings) -> list[StoreInfo]:
    """List store metadata for all directories under artifact root."""

    root = settings.artifact_root
    if not root.exists():
        return []

    stores: list[StoreInfo] = []
    for candidate in sorted(root.iterdir(), key=lambda path: path.name):
        if not candidate.is_dir():
            continue
        stores.append(_to_store_info(candidate))
    return stores


def get_store_info(store_name: str, settings: Settings) -> StoreInfo:
    """Return one store's metadata."""

    store_dir = _store_dir(store_name=store_name, settings=settings)
    if not store_dir.exists() or not store_dir.is_dir():
        raise IngestError(f"Store does not exist: {store_name}")
    return _to_store_info(store_dir)

