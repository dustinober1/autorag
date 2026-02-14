"""Typer CLI entrypoint."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from autokg_rag.answer import compose_grounded_answer
from autokg_rag.config import Settings, load_settings
from autokg_rag.exceptions import AutoRAGError
from autokg_rag.ingest import run_ingest_pipeline, run_smoke_pipeline
from autokg_rag.io import read_jsonl_rows, write_jsonl_rows
from autokg_rag.kg.pipeline import run_build_kg_pipeline, run_graph_query_pipeline
from autokg_rag.retrieval import run_hybrid_query_pipeline
from autokg_rag.schemas.api import AnswerPayload
from autokg_rag.schemas.records import HybridHitRecord
from autokg_rag.vector.pipeline import run_index_vector_pipeline, run_vector_query_pipeline
from autokg_rag.vector.store import load_chunks

app = typer.Typer(help="AutoRAG command line interface")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validated_run_id(raw_run_id: str) -> str:
    run_id = raw_run_id.strip()
    if not run_id:
        raise AutoRAGError("--run-id must be non-empty.")
    if ".." in run_id or not _RUN_ID_RE.fullmatch(run_id):
        raise AutoRAGError(
            "Invalid --run-id. Use letters, numbers, '.', '_' or '-' only, without '..'."
        )
    return run_id


def _run_hybrid_query(
    *,
    run_id: str,
    question: str,
    top_k: int,
    settings: Settings,
) -> list[HybridHitRecord]:
    return run_hybrid_query_pipeline(
        run_id=run_id,
        question=question,
        top_k=top_k,
        settings=settings,
    )


def _persist_answer_payload(payload: AnswerPayload, artifact_dir: Path) -> None:
    existing_answers = read_jsonl_rows(artifact_dir / "answers.jsonl")
    write_jsonl_rows(
        artifact_dir / "answers.jsonl",
        existing_answers + [payload.answer.model_dump(mode="json")],
    )

    existing_citation_trace = read_jsonl_rows(artifact_dir / "citation_trace.jsonl")
    write_jsonl_rows(
        artifact_dir / "citation_trace.jsonl",
        existing_citation_trace
        + [trace.model_dump(mode="json") for trace in payload.citation_trace],
    )


@app.callback()
def callback() -> None:
    """CLI root callback to keep command group semantics."""


@app.command()
def smoke(
    input_dir: Annotated[
        Path,
        typer.Option(
            ...,
            "--input",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    question: Annotated[str, typer.Option(..., "--question")],
    run_id: Annotated[str, typer.Option(..., "--run-id")],
) -> None:
    """Run Milestone 1 smoke pipeline and print answer payload."""

    try:
        run_id = _validated_run_id(run_id)
        settings = load_settings()
        answer = run_smoke_pipeline(
            input_dir=input_dir,
            question=question,
            run_id=run_id,
            settings=settings,
        )
    except (AutoRAGError, ValidationError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(answer.model_dump_json(indent=2))


@app.command()
def ingest(
    input_dir: Annotated[
        Path,
        typer.Option(
            ...,
            "--input",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    run_id: Annotated[str, typer.Option(..., "--run-id")],
    chunking: Annotated[str, typer.Option("--chunking")] = "heading_recursive",
) -> None:
    """Run Milestone 2 ingest pipeline and produce document/page/chunk artifacts."""

    try:
        run_id = _validated_run_id(run_id)
        settings = load_settings()
        doc_count, page_count, chunk_count = run_ingest_pipeline(
            input_dir=input_dir,
            run_id=run_id,
            chunking_strategy=chunking,
            settings=settings,
        )
    except (AutoRAGError, ValidationError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "documents": doc_count,
                "pages": page_count,
                "chunks": chunk_count,
                "chunking": chunking,
            },
            indent=2,
        )
    )


@app.command("index-vector")
def index_vector(
    run_id: Annotated[str, typer.Option(..., "--run-id")],
    embedding: Annotated[str, typer.Option(..., "--embedding")],
) -> None:
    """Build vector embedding artifacts from run chunks."""

    try:
        run_id = _validated_run_id(run_id)
        settings = load_settings()
        chunk_count = run_index_vector_pipeline(
            run_id=run_id,
            embedding_model=embedding,
            settings=settings,
        )
    except (AutoRAGError, ValidationError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "embedding_model": embedding,
                "indexed_chunks": chunk_count,
            },
            indent=2,
        )
    )


@app.command("build-kg")
def build_kg(run_id: Annotated[str, typer.Option(..., "--run-id")]) -> None:
    """Build knowledge-graph artifacts from chunked data."""

    try:
        run_id = _validated_run_id(run_id)
        settings = load_settings()
        node_count, edge_count, mention_count = run_build_kg_pipeline(
            run_id=run_id,
            settings=settings,
        )
    except (AutoRAGError, ValidationError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "nodes": node_count,
                "edges": edge_count,
                "chunk_mentions": mention_count,
            },
            indent=2,
        )
    )


@app.command()
def query(
    run_id: Annotated[str, typer.Option(..., "--run-id")],
    question: Annotated[str, typer.Option(..., "--question")],
    mode: Annotated[str, typer.Option("--mode")] = "vector",
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = 8,
) -> None:
    """Run retrieval query and print outputs."""

    try:
        run_id = _validated_run_id(run_id)
        settings = load_settings()

        if mode == "vector":
            vector_hits = run_vector_query_pipeline(
                run_id=run_id,
                question=question,
                top_k=top_k,
                settings=settings,
            )
            typer.echo(json.dumps([hit.model_dump(mode="json") for hit in vector_hits], indent=2))
            return

        if mode == "graph":
            hit_rows, answer = run_graph_query_pipeline(
                run_id=run_id,
                question=question,
                top_k=top_k,
                settings=settings,
            )
            typer.echo(
                json.dumps(
                    {
                        "answer": answer.model_dump(mode="json"),
                        "hits": hit_rows,
                    },
                    indent=2,
                )
            )
            return

        if mode == "hybrid":
            hybrid_hits = _run_hybrid_query(
                run_id=run_id,
                question=question,
                top_k=top_k,
                settings=settings,
            )
            typer.echo(json.dumps([hit.model_dump(mode="json") for hit in hybrid_hits], indent=2))
            return

        raise AutoRAGError(
            "Unsupported mode "
            f"'{mode}' for this milestone. Use --mode vector, --mode graph, or --mode hybrid."
        )
    except (AutoRAGError, ValidationError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


@app.command()
def answer(
    run_id: Annotated[str, typer.Option(..., "--run-id")],
    question: Annotated[str, typer.Option(..., "--question")],
    mode: Annotated[str, typer.Option("--mode")] = "hybrid",
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = 8,
) -> None:
    """Compose grounded answer payload and persist answer artifacts."""

    try:
        run_id = _validated_run_id(run_id)
        settings = load_settings()
        if mode != "hybrid":
            raise AutoRAGError(
                f"Unsupported mode '{mode}' for this milestone. Use --mode hybrid."
            )

        hits = _run_hybrid_query(
            run_id=run_id,
            question=question,
            top_k=top_k,
            settings=settings,
        )
        artifact_dir = settings.artifact_root / run_id
        chunk_by_id = {chunk.chunk_id: chunk for chunk in load_chunks(artifact_dir)}

        answer_record, citation_trace = compose_grounded_answer(
            question=question,
            hits=hits,
            chunk_by_id=chunk_by_id,
            max_sentences=3,
        )
        payload = AnswerPayload(
            answer=answer_record,
            hits=hits,
            citation_trace=citation_trace,
        )
        _persist_answer_payload(payload=payload, artifact_dir=artifact_dir)

        typer.echo(payload.model_dump_json(indent=2))
    except (AutoRAGError, ValidationError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


def main() -> None:
    """Console script entrypoint."""

    app()


if __name__ == "__main__":
    main()
