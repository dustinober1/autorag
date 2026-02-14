"""Typer CLI entrypoint."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from autokg_rag.answer import compose_grounded_answer
from autokg_rag.app_api import demo_build_endpoint, run_demo_doctor
from autokg_rag.config import Settings, load_settings
from autokg_rag.eval.dataset_builder import (
    bootstrap_starter_dataset,
    generate_dataset_from_chunks,
)
from autokg_rag.eval.matrix_runner import run_matrix
from autokg_rag.eval.report import build_experiment_report
from autokg_rag.exceptions import AutoRAGError
from autokg_rag.ingest import run_ingest_pipeline, run_smoke_pipeline
from autokg_rag.io import read_jsonl_rows, write_jsonl_rows
from autokg_rag.kg.pipeline import run_build_kg_pipeline, run_graph_query_pipeline
from autokg_rag.retrieval import run_hybrid_query_pipeline
from autokg_rag.schemas.api import AnswerPayload, QueryMode
from autokg_rag.schemas.records import HybridHitRecord
from autokg_rag.vector.pipeline import run_index_vector_pipeline, run_vector_query_pipeline
from autokg_rag.vector.store import load_chunks

app = typer.Typer(help="AutoRAG command line interface")
eval_app = typer.Typer(help="Evaluation harness commands")
app.add_typer(eval_app, name="eval")
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
    mode: Annotated[QueryMode, typer.Option("--mode")] = "hybrid",
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


@app.command()
def doctor(
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    input_dir: Annotated[Path | None, typer.Option("--input")] = None,
    reports_dir: Annotated[Path | None, typer.Option("--reports-dir")] = None,
    matrix_reports_dir: Annotated[Path | None, typer.Option("--matrix-reports-dir")] = None,
) -> None:
    """Validate local demo prerequisites and expected M6 artifacts."""

    try:
        raw_run_id = run_id if run_id is not None else os.getenv("AUTORAG_DEMO_RUN_ID")
        if raw_run_id is None:
            raw_run_id = "m6"
        resolved_run_id = _validated_run_id(raw_run_id)

        raw_input_dir = (
            str(input_dir)
            if input_dir is not None
            else os.getenv("AUTORAG_DEMO_INPUT_DIR", "data/fixtures/pdfs")
        )
        raw_reports_dir = (
            str(reports_dir)
            if reports_dir is not None
            else os.getenv("AUTORAG_DEMO_REPORTS_DIR", "reports/milestones")
        )
        raw_matrix_reports_dir = (
            str(matrix_reports_dir)
            if matrix_reports_dir is not None
            else os.getenv("AUTORAG_DEMO_MATRIX_REPORTS_DIR", "reports/experiments")
        )
        resolved_input_dir = Path(raw_input_dir)
        resolved_reports_dir = Path(raw_reports_dir)
        resolved_matrix_reports_dir = Path(raw_matrix_reports_dir)
        settings = load_settings()

        report = run_demo_doctor(
            run_id=resolved_run_id,
            input_dir=resolved_input_dir,
            artifact_root=settings.artifact_root,
            reports_dir=resolved_reports_dir,
            matrix_reports_dir=resolved_matrix_reports_dir,
        )
    except (AutoRAGError, ValidationError, OSError, ValueError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(report.model_dump_json(indent=2))
    if report.status == "error":
        for check in report.checks:
            if check.status != "missing":
                continue
            hint = check.hint or "Run `make demo-build`."
            typer.secho(
                f"[MISSING] {check.name}: {check.path} -> {hint}",
                err=True,
                fg=typer.colors.RED,
            )
        raise typer.Exit(code=1)


@app.command("demo-build")
def demo_build(
    run_id: Annotated[str, typer.Option("--run-id")] = "m6",
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = Path("data/fixtures/pdfs"),
    question: Annotated[
        str,
        typer.Option("--question"),
    ] = "Compare mitigation and acceptance strategies.",
    mode: Annotated[QueryMode, typer.Option("--mode")] = "hybrid",
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = 8,
    reports_dir: Annotated[Path, typer.Option("--reports-dir")] = Path("reports/milestones"),
    matrix_reports_dir: Annotated[
        Path,
        typer.Option("--matrix-reports-dir"),
    ] = Path("reports/experiments"),
) -> None:
    """Run the full Milestone 6 portfolio demo workflow."""

    try:
        run_id = _validated_run_id(run_id)
        settings = load_settings()
        summary = demo_build_endpoint(
            run_id=run_id,
            input_dir=input_dir,
            question=question,
            mode=mode,
            top_k=top_k,
            reports_dir=reports_dir,
            matrix_reports_dir=matrix_reports_dir,
            settings=settings,
        )
    except (AutoRAGError, ValidationError, OSError, ValueError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(summary, indent=2))


@eval_app.command("bootstrap-starter")
def eval_bootstrap_starter(
    out: Annotated[
        Path,
        typer.Option(
            ...,
            "--out",
            file_okay=True,
            dir_okay=False,
        ),
    ],
) -> None:
    """Write the embedded 20-question starter dataset JSONL."""

    try:
        rows = bootstrap_starter_dataset(out_path=out)
        typer.echo(
            json.dumps(
                {
                    "out": str(out),
                    "questions": len(rows),
                },
                indent=2,
            )
        )
    except (AutoRAGError, ValidationError, OSError, ValueError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


@eval_app.command("generate")
def eval_generate(
    run_id: Annotated[str, typer.Option(..., "--run-id")],
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
    target_size: Annotated[int, typer.Option(..., "--target-size", min=200, max=500)],
) -> None:
    """Generate evaluation questions from prior run artifacts."""

    try:
        run_id = _validated_run_id(run_id)
        settings = load_settings()
        output_path = generate_dataset_from_chunks(
            run_id=run_id,
            input_artifact_dir=input_dir,
            target_size=target_size,
            output_artifact_root=settings.artifact_root,
        )
        typer.echo(
            json.dumps(
                {
                    "run_id": run_id,
                    "target_size": target_size,
                    "output_path": str(output_path),
                    "questions": len(read_jsonl_rows(output_path)),
                },
                indent=2,
            )
        )
    except (AutoRAGError, ValidationError, OSError, ValueError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


@eval_app.command("run-matrix")
def eval_run_matrix(
    run_id: Annotated[str, typer.Option(..., "--run-id")],
    config: Annotated[
        Path,
        typer.Option(
            ...,
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
) -> None:
    """Execute factorial experiment matrix and persist reports."""

    try:
        run_id = _validated_run_id(run_id)
        settings = load_settings()
        rows = run_matrix(
            run_id=run_id,
            config_path=config,
            settings=settings,
        )
        typer.echo(
            json.dumps(
                {
                    "run_id": run_id,
                    "config": str(config),
                    "experiments": len(rows),
                    "reports_dir": "reports/experiments",
                    "matrix_results_csv": "reports/experiments/matrix_results.csv",
                    "matrix_results_json": "reports/experiments/matrix_results.json",
                    "per_query_dir": "reports/experiments/per_query",
                },
                indent=2,
            )
        )
    except (AutoRAGError, ValidationError, OSError, ValueError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


@eval_app.command("report")
def eval_report(run_id: Annotated[str, typer.Option(..., "--run-id")]) -> None:
    """Write markdown leaderboard report from matrix result artifacts."""

    try:
        run_id = _validated_run_id(run_id)
        summary = build_experiment_report(run_id=run_id)
        typer.echo(json.dumps(summary, indent=2))
    except (AutoRAGError, ValidationError, OSError, ValueError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


def main() -> None:
    """Console script entrypoint."""

    app()


if __name__ == "__main__":
    main()
