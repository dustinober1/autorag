"""Per-document management operations for vector stores."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, cast

import numpy as np

from autokg_rag.chunking import chunk_pages
from autokg_rag.chunking.base import ChunkingStrategy
from autokg_rag.config import Settings
from autokg_rag.embeddings.factory import create_embedding_provider
from autokg_rag.exceptions import IngestError, RetrievalError
from autokg_rag.ingest.pdf_parse import (
    extract_title,
    parse_pdf_pages_clean,
    sha256_for_file,
)
from autokg_rag.ingest.sectionize import detect_section
from autokg_rag.io import read_parquet_rows, write_parquet_rows
from autokg_rag.schemas.api import DocumentInfo, IngestResult
from autokg_rag.schemas.records import (
    ChunkRecord,
    DocumentRecord,
    EmbeddingMetaRecord,
    PageRecord,
)
from autokg_rag.vector.pipeline import run_index_vector_pipeline
from autokg_rag.vector.store import load_embedding_meta, load_embeddings


class UploadedFileLike(Protocol):
    """Protocol for Streamlit-like uploaded file wrappers."""

    name: str

    def getvalue(self) -> bytes:
        """Return full file bytes."""


def _artifact_dir(*, store_name: str, settings: Settings) -> Path:
    normalized = store_name.strip()
    if not normalized:
        raise IngestError("Store name must not be empty.")
    return settings.artifact_root / normalized


def _load_documents(path: Path) -> list[DocumentRecord]:
    rows = read_parquet_rows(path / "documents.parquet")
    return [DocumentRecord.model_validate(row) for row in rows]


def _load_pages(path: Path) -> list[PageRecord]:
    rows = read_parquet_rows(path / "pages.parquet")
    return [PageRecord.model_validate(row) for row in rows]


def _load_chunks(path: Path) -> list[ChunkRecord]:
    rows = read_parquet_rows(path / "chunks.parquet")
    return [ChunkRecord.model_validate(row) for row in rows]


def _write_store_records(
    *,
    artifact_dir: Path,
    documents: list[DocumentRecord],
    pages: list[PageRecord],
    chunks: list[ChunkRecord],
) -> None:
    write_parquet_rows(
        artifact_dir / "documents.parquet",
        [row.model_dump(mode="json") for row in documents],
    )
    write_parquet_rows(
        artifact_dir / "pages.parquet",
        [row.model_dump(mode="json") for row in pages],
    )
    write_parquet_rows(
        artifact_dir / "chunks.parquet",
        [row.model_dump(mode="json") for row in chunks],
    )


def _resolve_chunking_strategy(chunks: list[ChunkRecord]) -> ChunkingStrategy:
    if not chunks:
        return "heading_recursive"

    candidate = chunks[0].chunk_id
    for strategy in ("heading_recursive", "sentence_window", "semantic_breakpoint", "fixed"):
        if f"-{strategy}-" in candidate:
            return cast(ChunkingStrategy, strategy)
    return "heading_recursive"


def _read_uploaded_bytes(uploaded: UploadedFileLike | Any) -> bytes:
    getter = getattr(uploaded, "getvalue", None)
    if callable(getter):
        data = getter()
        if isinstance(data, bytes):
            return data
        if isinstance(data, bytearray):
            return bytes(data)
        raise IngestError("Uploaded file getvalue() must return bytes.")

    reader = getattr(uploaded, "read", None)
    if callable(reader):
        data = reader()
        if hasattr(uploaded, "seek"):
            try:
                uploaded.seek(0)
            except OSError:
                pass
        if isinstance(data, bytes):
            return data
        if isinstance(data, bytearray):
            return bytes(data)
        raise IngestError("Uploaded file read() must return bytes.")

    raise IngestError("Unsupported uploaded file object; expected getvalue() or read().")


def _sanitize_upload_name(name: str, fallback_index: int) -> str:
    normalized = Path(name).name.strip()
    if not normalized:
        normalized = f"upload_{fallback_index}.pdf"
    if not normalized.lower().endswith(".pdf"):
        normalized = f"{normalized}.pdf"
    return normalized


def _save_uploads_to_temp(files: Sequence[UploadedFileLike | Any], temp_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for index, uploaded in enumerate(files, start=1):
        filename = _sanitize_upload_name(str(getattr(uploaded, "name", "")), index)
        payload = _read_uploaded_bytes(uploaded)
        output_path = temp_dir / filename
        output_path.write_bytes(payload)
        paths.append(output_path)
    return paths


def _append_embeddings(
    *,
    artifact_dir: Path,
    new_chunks: list[ChunkRecord],
    settings: Settings,
) -> None:
    if not new_chunks:
        return

    meta_rows = load_embedding_meta(artifact_dir)
    matrix = load_embeddings(artifact_dir)
    if matrix.ndim != 2:
        raise RetrievalError("Embedding matrix must be two-dimensional.")

    ordered_meta = sorted(meta_rows, key=lambda row: row.row_idx)
    if int(matrix.shape[0]) != len(ordered_meta):
        raise RetrievalError("Embedding rows and metadata count do not match.")

    if not ordered_meta:
        run_index_vector_pipeline(
            run_id=artifact_dir.name,
            embedding_model=settings.embedding_model,
            settings=settings,
        )
        return

    provider_name = ordered_meta[0].provider
    embedding_model = ordered_meta[0].embedding_model
    dim = int(ordered_meta[0].dim)

    provider_selection = create_embedding_provider(
        embedding_provider=provider_name,
        embedding_model=embedding_model,
        embedding_dim=dim,
        ollama_base_url=settings.ollama_base_url,
        ollama_timeout_seconds=settings.ollama_timeout_seconds,
    )
    new_matrix = provider_selection.provider.embed_texts([chunk.chunk_text for chunk in new_chunks])
    if new_matrix.ndim != 2:
        raise RetrievalError("Embedding provider returned a non-2D matrix for new chunks.")
    if int(new_matrix.shape[0]) != len(new_chunks):
        raise RetrievalError("Embedding provider returned unexpected row count for new chunks.")
    if int(new_matrix.shape[1]) != dim:
        raise RetrievalError(
            "Embedding provider returned unexpected dimensionality for incremental chunks."
        )

    combined = np.concatenate([matrix, new_matrix.astype(np.float32)], axis=0)
    np.save(artifact_dir / "embeddings.npy", combined.astype(np.float32))

    next_meta_rows = [row.model_dump(mode="json") for row in ordered_meta]
    start = len(next_meta_rows)
    next_meta_rows.extend(
        EmbeddingMetaRecord(
            chunk_id=chunk.chunk_id,
            row_idx=start + offset,
            provider=provider_name,
            embedding_model=embedding_model,
            dim=dim,
        ).model_dump(mode="json")
        for offset, chunk in enumerate(new_chunks)
    )
    write_parquet_rows(artifact_dir / "embedding_meta.parquet", next_meta_rows)


def _rewrite_embeddings_after_chunk_filter(
    *,
    artifact_dir: Path,
    remaining_chunk_ids: set[str],
) -> None:
    embeddings_path = artifact_dir / "embeddings.npy"
    meta_path = artifact_dir / "embedding_meta.parquet"
    if not embeddings_path.exists() or not meta_path.exists():
        return

    meta_rows = sorted(load_embedding_meta(artifact_dir), key=lambda row: row.row_idx)
    matrix = load_embeddings(artifact_dir)
    if matrix.ndim != 2:
        raise RetrievalError("Embedding matrix must be two-dimensional.")
    if int(matrix.shape[0]) != len(meta_rows):
        raise RetrievalError("Embedding rows and metadata count do not match.")

    kept_rows = [row for row in meta_rows if row.chunk_id in remaining_chunk_ids]
    kept_indices = [row.row_idx for row in kept_rows]
    if any(index < 0 or index >= int(matrix.shape[0]) for index in kept_indices):
        raise RetrievalError("Embedding metadata contains out-of-range row indices.")

    if kept_indices:
        kept_matrix = matrix[kept_indices, :]
    else:
        dim = int(matrix.shape[1])
        kept_matrix = np.zeros((0, dim), dtype=np.float32)

    np.save(embeddings_path, kept_matrix.astype(np.float32))
    next_meta_rows = [
        row.model_copy(update={"row_idx": idx}).model_dump(mode="json")
        for idx, row in enumerate(kept_rows)
    ]
    write_parquet_rows(meta_path, next_meta_rows)


def _build_records_from_pdf_paths(
    *,
    pdf_paths: Sequence[Path],
    existing_sha256: set[str],
    chunking_strategy: ChunkingStrategy,
    settings: Settings,
) -> tuple[list[DocumentRecord], list[PageRecord], list[ChunkRecord], int]:
    new_documents: list[DocumentRecord] = []
    new_pages: list[PageRecord] = []
    skipped_duplicates = 0

    seen_sha_in_batch: set[str] = set()
    for pdf_path in pdf_paths:
        sha = sha256_for_file(pdf_path)
        if sha in existing_sha256 or sha in seen_sha_in_batch:
            skipped_duplicates += 1
            continue

        seen_sha_in_batch.add(sha)
        doc_id = f"doc_{sha[:12]}"
        page_texts = parse_pdf_pages_clean(pdf_path)
        title = extract_title(page_texts, fallback=pdf_path.stem)

        new_documents.append(
            DocumentRecord(
                doc_id=doc_id,
                title=title,
                source_path=str(pdf_path),
                sha256=sha,
            )
        )
        for page_number, page_text in enumerate(page_texts, start=1):
            new_pages.append(
                PageRecord(
                    doc_id=doc_id,
                    page=page_number,
                    section=detect_section(page_text),
                    text=page_text,
                )
            )

    new_chunks = chunk_pages(
        pages=new_pages,
        strategy=chunking_strategy,
        chunk_word_size=settings.chunk_word_size,
        chunk_word_overlap=settings.chunk_word_overlap,
        sentence_window_size=settings.sentence_window_size,
        semantic_similarity_breakpoint=settings.semantic_similarity_breakpoint,
    )
    return new_documents, new_pages, new_chunks, skipped_duplicates


def list_documents(store_name: str, settings: Settings) -> list[DocumentInfo]:
    """List documents in a store with page/chunk counts."""

    artifact_dir = _artifact_dir(store_name=store_name, settings=settings)
    documents = _load_documents(artifact_dir)
    pages = _load_pages(artifact_dir)
    chunks = _load_chunks(artifact_dir)

    page_counts = Counter(page.doc_id for page in pages)
    chunk_counts = Counter(chunk.doc_id for chunk in chunks)

    info_rows = [
        DocumentInfo(
            doc_id=document.doc_id,
            title=document.title,
            source_path=document.source_path,
            sha256=document.sha256,
            page_count=int(page_counts.get(document.doc_id, 0)),
            chunk_count=int(chunk_counts.get(document.doc_id, 0)),
        )
        for document in documents
    ]
    return sorted(info_rows, key=lambda row: row.title.lower())


def remove_document(store_name: str, doc_id: str, settings: Settings) -> bool:
    """Remove one document and rewrite dependent chunk + embedding artifacts."""

    normalized_doc_id = doc_id.strip()
    if not normalized_doc_id:
        raise IngestError("doc_id must not be empty.")

    artifact_dir = _artifact_dir(store_name=store_name, settings=settings)
    documents = _load_documents(artifact_dir)
    if not any(document.doc_id == normalized_doc_id for document in documents):
        return False

    pages = _load_pages(artifact_dir)
    chunks = _load_chunks(artifact_dir)

    next_documents = [row for row in documents if row.doc_id != normalized_doc_id]
    next_pages = [row for row in pages if row.doc_id != normalized_doc_id]
    next_chunks = [row for row in chunks if row.doc_id != normalized_doc_id]

    _write_store_records(
        artifact_dir=artifact_dir,
        documents=next_documents,
        pages=next_pages,
        chunks=next_chunks,
    )
    _rewrite_embeddings_after_chunk_filter(
        artifact_dir=artifact_dir,
        remaining_chunk_ids={chunk.chunk_id for chunk in next_chunks},
    )
    return True


def add_document_paths(
    store_name: str,
    pdf_paths: Sequence[Path],
    settings: Settings,
) -> IngestResult:
    """Incrementally ingest PDF paths into an existing store."""

    if not pdf_paths:
        raise IngestError("No input files provided.")

    artifact_dir = _artifact_dir(store_name=store_name, settings=settings)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    existing_documents = _load_documents(artifact_dir)
    existing_pages = _load_pages(artifact_dir)
    existing_chunks = _load_chunks(artifact_dir)

    chunking_strategy = _resolve_chunking_strategy(existing_chunks)
    existing_sha = {document.sha256 for document in existing_documents}
    new_documents, new_pages, new_chunks, skipped_duplicates = _build_records_from_pdf_paths(
        pdf_paths=pdf_paths,
        existing_sha256=existing_sha,
        chunking_strategy=chunking_strategy,
        settings=settings,
    )

    if not new_documents:
        return IngestResult(
            store_name=artifact_dir.name,
            documents=0,
            pages=0,
            chunks=0,
            skipped_duplicates=skipped_duplicates,
        )

    combined_documents = [*existing_documents, *new_documents]
    combined_pages = [*existing_pages, *new_pages]
    combined_chunks = [*existing_chunks, *new_chunks]
    _write_store_records(
        artifact_dir=artifact_dir,
        documents=combined_documents,
        pages=combined_pages,
        chunks=combined_chunks,
    )

    embeddings_path = artifact_dir / "embeddings.npy"
    embedding_meta_path = artifact_dir / "embedding_meta.parquet"
    if embeddings_path.exists() and embedding_meta_path.exists() and existing_chunks:
        _append_embeddings(
            artifact_dir=artifact_dir,
            new_chunks=new_chunks,
            settings=settings,
        )
    else:
        run_index_vector_pipeline(
            run_id=artifact_dir.name,
            embedding_model=settings.embedding_model,
            settings=settings,
        )

    return IngestResult(
        store_name=artifact_dir.name,
        documents=len(new_documents),
        pages=len(new_pages),
        chunks=len(new_chunks),
        skipped_duplicates=skipped_duplicates,
    )


def add_documents(
    store_name: str,
    files: Sequence[UploadedFileLike | Any],
    settings: Settings,
) -> IngestResult:
    """Incrementally ingest uploaded files into a store."""

    if not files:
        raise IngestError("No uploaded files provided.")

    with TemporaryDirectory(prefix="autorag-upload-") as raw_temp_dir:
        temp_dir = Path(raw_temp_dir)
        pdf_paths = _save_uploads_to_temp(files, temp_dir)
        return add_document_paths(store_name, pdf_paths, settings)

