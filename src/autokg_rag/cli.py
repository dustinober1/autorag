"""Typer CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from autokg_rag.config import load_settings
from autokg_rag.exceptions import AutoRAGError
from autokg_rag.ingest import run_ingest_pipeline, run_smoke_pipeline
from autokg_rag.vector.pipeline import run_index_vector_pipeline, run_vector_query_pipeline

app = typer.Typer(help="AutoRAG command line interface")


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


@app.command()
def query(
    run_id: Annotated[str, typer.Option(..., "--run-id")],
    question: Annotated[str, typer.Option(..., "--question")],
    mode: Annotated[str, typer.Option("--mode")] = "vector",
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = 8,
) -> None:
    """Run retrieval query and print hits."""

    try:
        if mode != "vector":
            raise AutoRAGError(f"Unsupported mode '{mode}' for this milestone. Use --mode vector.")

        settings = load_settings()
        hits = run_vector_query_pipeline(
            run_id=run_id,
            question=question,
            top_k=top_k,
            settings=settings,
        )
    except (AutoRAGError, ValidationError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps([hit.model_dump(mode="json") for hit in hits], indent=2))


def main() -> None:
    """Console script entrypoint."""

    app()


if __name__ == "__main__":
    main()
