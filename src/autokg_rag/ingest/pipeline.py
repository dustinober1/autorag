"""Ingest pipelines for M1 smoke and M2 vector baseline."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from autokg_rag.chunking import SUPPORTED_CHUNKING_STRATEGIES, chunk_pages
from autokg_rag.chunking.base import ChunkingStrategy
from autokg_rag.chunking.fixed import chunk_page
from autokg_rag.config import Settings, write_resolved_config
from autokg_rag.exceptions import IngestError, RetrievalError
from autokg_rag.ingest.manifest import build_raw_documents
from autokg_rag.ingest.pdf_parse import (
    discover_pdf_files,
    extract_title,
    parse_pdf_pages_clean,
    sha256_for_file,
)
from autokg_rag.ingest.sectionize import detect_section
from autokg_rag.io import write_jsonl_rows, write_parquet_rows
from autokg_rag.observability import MetricsWriter, StructuredLogger
from autokg_rag.schemas.provenance import Citation
from autokg_rag.schemas.records import (
    AnswerRecord,
    ChunkRecord,
    DocumentRecord,
    PageRecord,
)
from autokg_rag.vector.retriever import retrieve_top_chunks


def _compose_answer_text(question: str, best_chunk: ChunkRecord) -> str:
    sentence = best_chunk.chunk_text.strip()
    if len(sentence) > 280:
        sentence = f"{sentence[:277]}..."
    return f"For '{question}', the source states: {sentence}"


def run_smoke_pipeline(
    input_dir: Path,
    question: str,
    run_id: str,
    settings: Settings,
) -> AnswerRecord:
    """Execute Milestone 1 smoke workflow and persist required artifacts."""

    artifact_dir = settings.artifact_root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    write_resolved_config(settings, artifact_dir / "resolved_config.yaml")

    logger = StructuredLogger(run_id=run_id, output_path=artifact_dir / "logs.jsonl")
    metrics = MetricsWriter(run_id=run_id, output_path=artifact_dir / "metrics.jsonl")

    logger.info(stage="pipeline", event="start", question=question, input_dir=str(input_dir))

    with metrics.timer(stage="ingest", metric_name="ingest.seconds"):
        logger.info(stage="ingest", event="start")
        raw_documents = build_raw_documents(input_dir=input_dir)
        logger.info(stage="ingest", event="complete", documents=len(raw_documents))
        metrics.counter(
            stage="ingest",
            metric_name="documents.count",
            value=float(len(raw_documents)),
        )

    manifest_payloads = [doc.manifest.model_dump(mode="json") for doc in raw_documents]
    write_jsonl_rows(artifact_dir / "doc_manifest.jsonl", manifest_payloads)

    with metrics.timer(stage="chunking", metric_name="chunking.seconds"):
        logger.info(stage="chunking", event="start")
        chunks: list[ChunkRecord] = []
        for document in raw_documents:
            for page_number, page_text in enumerate(document.pages, start=1):
                section = detect_section(page_text)
                chunks.extend(
                    chunk_page(
                        doc_id=document.manifest.doc_id,
                        page=page_number,
                        section=section,
                        text=page_text,
                        chunk_word_size=settings.chunk_word_size,
                        chunk_word_overlap=settings.chunk_word_overlap,
                    )
                )
        logger.info(stage="chunking", event="complete", chunks=len(chunks))
        metrics.counter(stage="chunking", metric_name="chunks.count", value=float(len(chunks)))

    write_jsonl_rows(
        artifact_dir / "chunks.jsonl",
        [chunk.model_dump(mode="json") for chunk in chunks],
    )

    with metrics.timer(stage="retrieval", metric_name="retrieval.seconds"):
        logger.info(stage="retrieval", event="start", top_k=settings.top_k)
        hits = retrieve_top_chunks(question=question, chunks=chunks, top_k=settings.top_k)
        if not hits:
            raise RetrievalError("Retriever returned no chunks")
        best = hits[0].chunk
        logger.info(
            stage="retrieval",
            event="complete",
            hits=len(hits),
            best_chunk_id=best.chunk_id,
        )

    citation = Citation(
        chunk_id=best.chunk_id,
        doc_id=best.doc_id,
        page=best.page,
        section=best.section,
    )
    answer = AnswerRecord(
        question_id=f"{run_id}:q1",
        answer_text=_compose_answer_text(question=question, best_chunk=best),
        citations=[citation],
    )

    with metrics.timer(stage="answer", metric_name="answer.seconds"):
        logger.info(stage="answer", event="start")
        (artifact_dir / "answer.json").write_text(
            answer.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info(stage="answer", event="complete", citations=len(answer.citations))
        metrics.counter(
            stage="answer",
            metric_name="citations.count",
            value=float(len(answer.citations)),
        )

    logger.info(stage="pipeline", event="complete", artifact_dir=str(artifact_dir))
    return answer


def run_ingest_pipeline(
    input_dir: Path,
    run_id: str,
    chunking_strategy: str,
    settings: Settings,
) -> tuple[int, int, int]:
    """Execute Milestone 2 ingest and chunking pipeline."""

    if chunking_strategy not in SUPPORTED_CHUNKING_STRATEGIES:
        supported = ", ".join(SUPPORTED_CHUNKING_STRATEGIES)
        raise IngestError(
            f"Unsupported chunking strategy '{chunking_strategy}'. "
            f"Expected one of: {supported}"
        )

    strategy = cast(ChunkingStrategy, chunking_strategy)
    artifact_dir = settings.artifact_root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    write_resolved_config(settings, artifact_dir / "resolved_config.yaml")

    logger = StructuredLogger(run_id=run_id, output_path=artifact_dir / "logs.jsonl")
    metrics = MetricsWriter(run_id=run_id, output_path=artifact_dir / "metrics.jsonl")

    logger.info(
        stage="pipeline",
        event="start",
        input_dir=str(input_dir),
        chunking_strategy=strategy,
    )

    documents: list[DocumentRecord] = []
    pages: list[PageRecord] = []

    with metrics.timer(stage="ingest", metric_name="ingest.seconds"):
        logger.info(stage="ingest", event="start")
        pdf_files = discover_pdf_files(input_dir)
        for file_path in pdf_files:
            sha = sha256_for_file(file_path)
            doc_id = f"doc_{sha[:12]}"

            page_texts = parse_pdf_pages_clean(file_path)
            title = extract_title(page_texts, fallback=file_path.stem)

            document = DocumentRecord(
                doc_id=doc_id,
                title=title,
                source_path=str(file_path),
                sha256=sha,
            )
            documents.append(document)

            for page_number, page_text in enumerate(page_texts, start=1):
                pages.append(
                    PageRecord(
                        doc_id=doc_id,
                        page=page_number,
                        section=detect_section(page_text),
                        text=page_text,
                    )
                )

        logger.info(
            stage="ingest",
            event="complete",
            documents=len(documents),
            pages=len(pages),
        )
        metrics.counter(stage="ingest", metric_name="documents.count", value=float(len(documents)))
        metrics.counter(stage="ingest", metric_name="pages.count", value=float(len(pages)))

    write_parquet_rows(
        artifact_dir / "documents.parquet",
        [doc.model_dump(mode="json") for doc in documents],
    )
    write_parquet_rows(
        artifact_dir / "pages.parquet",
        [page.model_dump(mode="json") for page in pages],
    )

    with metrics.timer(stage="chunking", metric_name="chunking.seconds"):
        logger.info(stage="chunking", event="start", strategy=strategy)
        chunks = chunk_pages(
            pages=pages,
            strategy=strategy,
            chunk_word_size=settings.chunk_word_size,
            chunk_word_overlap=settings.chunk_word_overlap,
            sentence_window_size=settings.sentence_window_size,
            semantic_similarity_breakpoint=settings.semantic_similarity_breakpoint,
        )
        logger.info(stage="chunking", event="complete", chunks=len(chunks))
        metrics.counter(stage="chunking", metric_name="chunks.count", value=float(len(chunks)))

    write_parquet_rows(
        artifact_dir / "chunks.parquet",
        [chunk.model_dump(mode="json") for chunk in chunks],
    )

    logger.info(stage="pipeline", event="complete", artifact_dir=str(artifact_dir))
    return len(documents), len(pages), len(chunks)
