from __future__ import annotations

from pathlib import Path

from autokg_rag.config.settings import Settings
from autokg_rag.ingest.pdf_parse import parse_pdf_pages_clean
from autokg_rag.ingest.pipeline import run_ingest_pipeline
from autokg_rag.io import read_parquet_rows
from autokg_rag.schemas.records import DocumentRecord


def _write_text_pdf(path: Path, title: str, body: str) -> None:
    path.write_text(f"{title}\n{body}\n", encoding="utf-8")


def test_pdf_ingest_extracts_pages_sections_and_stable_doc_ids(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdfs"
    input_dir.mkdir(parents=True)

    _write_text_pdf(
        input_dir / "scope_a.pdf",
        "Project Scope",
        "Scope baseline controls approved changes and expectation alignment.\f"
        "Execution Notes\nExecution requires change tracking.",
    )
    _write_text_pdf(
        input_dir / "scope_b.pdf",
        "Risk Response",
        "Mitigation and acceptance are documented with evidence.",
    )

    settings = Settings(artifact_root=tmp_path / "artifacts")

    run_ingest_pipeline(
        input_dir=input_dir,
        run_id="m2_first",
        chunking_strategy="heading_recursive",
        settings=settings,
    )
    run_ingest_pipeline(
        input_dir=input_dir,
        run_id="m2_second",
        chunking_strategy="heading_recursive",
        settings=settings,
    )

    docs_first = read_parquet_rows(settings.artifact_root / "m2_first" / "documents.parquet")
    docs_second = read_parquet_rows(settings.artifact_root / "m2_second" / "documents.parquet")

    doc_ids_first = {row["source_path"]: row["doc_id"] for row in docs_first}
    doc_ids_second = {row["source_path"]: row["doc_id"] for row in docs_second}
    assert doc_ids_first == doc_ids_second
    assert all(str(row.get("document_type", "")) == "generic" for row in docs_first)

    pages = read_parquet_rows(settings.artifact_root / "m2_first" / "pages.parquet")
    assert pages
    assert all(str(row["text"]).strip() != "" for row in pages)
    assert all(str(row["section"]).strip() != "" for row in pages)


def test_parse_pdf_pages_clean_removes_repeated_lines(tmp_path: Path) -> None:
    path = tmp_path / "repeated.pdf"
    path.write_text(
        "PMBOK Guide\nUnique A\fPMBOK Guide\nUnique B\fPMBOK Guide\nUnique C",
        encoding="utf-8",
    )

    pages = parse_pdf_pages_clean(path)
    assert len(pages) == 3
    assert all("PMBOK Guide" not in page for page in pages)


def test_document_record_defaults_document_type_for_backward_compatibility() -> None:
    record = DocumentRecord.model_validate(
        {
            "doc_id": "doc_123",
            "title": "Legacy document",
            "source_path": "legacy.pdf",
            "sha256": "a" * 64,
        }
    )
    assert record.document_type == "generic"
