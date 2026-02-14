"""Milestone 3 knowledge-graph build and query pipelines."""

from __future__ import annotations

from autokg_rag.config import Settings
from autokg_rag.exceptions import RetrievalError
from autokg_rag.io import read_jsonl_rows, write_jsonl_rows, write_parquet_rows
from autokg_rag.kg.ontology_extract import extract_ontology_from_chunks
from autokg_rag.kg.retriever import retrieve_graph_hits
from autokg_rag.kg.store_sqlite import persist_graph_sqlite
from autokg_rag.observability import MetricsWriter, StructuredLogger
from autokg_rag.schemas.provenance import Citation
from autokg_rag.schemas.records import AnswerRecord
from autokg_rag.vector.store import load_chunks


def run_build_kg_pipeline(run_id: str, settings: Settings) -> tuple[int, int, int]:
    """Build KG artifacts from chunk parquet for a run id."""

    artifact_dir = settings.artifact_root / run_id
    logger = StructuredLogger(run_id=run_id, output_path=artifact_dir / "logs.jsonl")
    metrics = MetricsWriter(run_id=run_id, output_path=artifact_dir / "metrics.jsonl")

    with metrics.timer(stage="build_kg", metric_name="build_kg.seconds"):
        chunks = load_chunks(artifact_dir)
        if not chunks:
            raise RetrievalError("No chunks found. Run ingest before build-kg.")

        nodes, edges, mentions = extract_ontology_from_chunks(chunks)

        write_parquet_rows(
            artifact_dir / "kg_nodes.parquet",
            [node.model_dump(mode="json") for node in nodes],
        )
        write_parquet_rows(
            artifact_dir / "kg_edges.parquet",
            [edge.model_dump(mode="json") for edge in edges],
        )

        persist_graph_sqlite(
            sqlite_path=artifact_dir / "kg.sqlite",
            nodes=nodes,
            edges=edges,
            chunk_mentions=mentions,
        )

        logger.info(
            stage="build_kg",
            event="complete",
            nodes=len(nodes),
            edges=len(edges),
            mentions=len(mentions),
        )
        metrics.counter(stage="build_kg", metric_name="kg.nodes", value=float(len(nodes)))
        metrics.counter(stage="build_kg", metric_name="kg.edges", value=float(len(edges)))

    return len(nodes), len(edges), len(mentions)


def _compose_graph_answer(question: str, supporting_chunk_text: str) -> str:
    snippet = supporting_chunk_text.strip()
    if len(snippet) > 280:
        snippet = f"{snippet[:277]}..."
    return f"For '{question}', graph evidence indicates: {snippet}"


def run_graph_query_pipeline(
    *,
    run_id: str,
    question: str,
    top_k: int,
    settings: Settings,
) -> tuple[list[dict[str, object]], AnswerRecord]:
    """Run graph retrieval, persist graph hits, and write a cited answer payload."""

    artifact_dir = settings.artifact_root / run_id
    logger = StructuredLogger(run_id=run_id, output_path=artifact_dir / "logs.jsonl")
    metrics = MetricsWriter(run_id=run_id, output_path=artifact_dir / "metrics.jsonl")

    with metrics.timer(stage="query_graph", metric_name="query_graph.seconds"):
        hits = retrieve_graph_hits(
            run_id=run_id,
            question=question,
            artifact_dir=artifact_dir,
            top_k=top_k,
            max_depth=settings.graph_max_depth,
        )
        if not hits:
            raise RetrievalError("Graph retrieval returned no hits.")

        existing_rows = read_jsonl_rows(artifact_dir / "graph_hits.jsonl")
        hit_rows = [hit.model_dump(mode="json") for hit in hits]
        write_jsonl_rows(artifact_dir / "graph_hits.jsonl", existing_rows + hit_rows)

        chunk_rows = load_chunks(artifact_dir)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunk_rows}

        citations = [
            Citation(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                page=hit.page,
                section=hit.section,
            )
            for hit in hits
        ]

        best_chunk = chunk_by_id.get(hits[0].chunk_id)
        if best_chunk is None:
            raise RetrievalError(f"Graph hit references unknown chunk_id: {hits[0].chunk_id}")

        answer = AnswerRecord(
            question_id=hits[0].question_id,
            answer_text=_compose_graph_answer(question, best_chunk.chunk_text),
            citations=citations,
        )
        (artifact_dir / "answer.json").write_text(
            answer.model_dump_json(indent=2),
            encoding="utf-8",
        )

        logger.info(
            stage="query_graph",
            event="complete",
            question_id=hits[0].question_id,
            hits=len(hits),
            citations=len(citations),
        )
        metrics.counter(stage="query_graph", metric_name="hits.count", value=float(len(hits)))

    return hit_rows, answer
