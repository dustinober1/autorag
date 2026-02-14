"""Typer CLI entrypoint."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, cast

import typer
from pydantic import ValidationError

from autokg_rag.answer import OllamaSentenceAdapter, compose_grounded_answer
from autokg_rag.app_api import (
    demo_build_endpoint,
    query_service,
    run_artifact_retention,
    run_demo_doctor,
)
from autokg_rag.config import Settings, load_settings
from autokg_rag.eval.ab_test import RetrievalMode, format_ab_test_report, run_ab_test
from autokg_rag.eval.dataset_builder import (
    bootstrap_starter_dataset,
    generate_dataset_from_chunks,
)
from autokg_rag.eval.judge import JudgementCriteria, evaluate_with_llm_judge
from autokg_rag.eval.matrix_runner import run_matrix
from autokg_rag.eval.metrics import ndcg_at_k, recall_at_k
from autokg_rag.eval.report import build_experiment_report
from autokg_rag.exceptions import AutoRAGError
from autokg_rag.ingest import run_ingest_pipeline, run_smoke_pipeline
from autokg_rag.io import read_jsonl_rows, write_jsonl_rows
from autokg_rag.kg.pipeline import run_build_kg_pipeline, run_graph_query_pipeline
from autokg_rag.retrieval import run_hybrid_query_pipeline
from autokg_rag.schemas.api import AnswerPayload, QueryMode, QueryRequest
from autokg_rag.schemas.records import EvalQuestionRecord, HybridHitRecord, RetrievalHitRecord
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


def _coerce_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


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


def _load_eval_questions(path: Path) -> list[EvalQuestionRecord]:
    rows = read_jsonl_rows(path)
    questions = [EvalQuestionRecord.model_validate(row) for row in rows]
    if not questions:
        raise AutoRAGError(f"Evaluation questions file is empty: {path}")
    return questions


def _context_for_hits(
    *,
    hits: list[HybridHitRecord] | list[RetrievalHitRecord],
    chunk_by_id: Mapping[str, object],
) -> list[str]:
    context: list[str] = []
    for hit in hits:
        chunk = chunk_by_id.get(hit.chunk_id)
        if chunk is None:
            continue
        chunk_text = getattr(chunk, "chunk_text", "")
        if isinstance(chunk_text, str) and chunk_text.strip():
            context.append(chunk_text)
    return context


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
    use_local: Annotated[bool, typer.Option("--use-local/--no-use-local")] = False,
    local_model: Annotated[str | None, typer.Option("--local-model")] = None,
    local_temperature: Annotated[float | None, typer.Option("--local-temperature")] = None,
    local_max_tokens: Annotated[int | None, typer.Option("--local-max-tokens")] = None,
    max_sentences: Annotated[int | None, typer.Option("--max-sentences", min=1)] = None,
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

        sentence_adapter = None
        configured_max_sentences = max(1, int(getattr(settings, "answer_max_sentences", 6)))
        resolved_max_sentences = (
            configured_max_sentences if max_sentences is None else max(1, int(max_sentences))
        )
        resolved_use_local = use_local or bool(getattr(settings, "answer_use_local", False))
        if resolved_use_local:
            configured_model = str(getattr(settings, "answer_model", "llama3")).strip() or "llama3"
            configured_temperature = float(getattr(settings, "answer_temperature", 0.2))
            configured_max_tokens = max(1, int(getattr(settings, "answer_max_tokens", 512)))
            resolved_model = (
                local_model.strip()
                if isinstance(local_model, str) and local_model.strip()
                else configured_model
            )
            sentence_adapter = OllamaSentenceAdapter(
                model=resolved_model,
                temperature=(
                    configured_temperature
                    if local_temperature is None
                    else float(local_temperature)
                ),
                max_tokens=(
                    configured_max_tokens
                    if local_max_tokens is None
                    else max(1, int(local_max_tokens))
                ),
                ollama_base_url=settings.ollama_base_url,
                ollama_timeout_seconds=settings.ollama_timeout_seconds,
            )

        answer_record, citation_trace = compose_grounded_answer(
            question=question,
            hits=hits,
            chunk_by_id=chunk_by_id,
            max_sentences=resolved_max_sentences,
            sentence_adapter=sentence_adapter,
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
            settings=settings,
        )
    except (AutoRAGError, ValidationError, OSError, ValueError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(report.model_dump_json(indent=2))
    if report.status == "error":
        for check in report.checks:
            if check.status == "valid":
                continue
            hint = check.hint or "Run `make demo-build`."
            label = check.status.upper()
            typer.secho(
                f"[{label}] {check.name}: {check.path} -> {hint}",
                err=True,
                fg=typer.colors.RED,
            )
        raise typer.Exit(code=1)


@app.command("cleanup-artifacts")
def cleanup_artifacts(
    keep_latest: Annotated[int, typer.Option("--keep-latest", min=0)] = 3,
    keep_run: Annotated[list[str] | None, typer.Option("--keep-run")] = None,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Delete candidate run directories. Defaults to dry-run when omitted.",
        ),
    ] = False,
    artifact_root: Annotated[Path | None, typer.Option("--artifact-root")] = None,
) -> None:
    """Plan or apply artifact retention cleanup (dry-run by default)."""

    try:
        settings = load_settings()
        resolved_artifact_root = settings.artifact_root if artifact_root is None else artifact_root
        resolved_keep_run_ids: set[str] = set()
        for candidate in keep_run or []:
            resolved_keep_run_ids.add(_validated_run_id(candidate))

        report = run_artifact_retention(
            artifact_root=resolved_artifact_root,
            keep_latest=keep_latest,
            keep_run_ids=resolved_keep_run_ids,
            dry_run=not apply,
        )
    except (AutoRAGError, ValidationError, OSError, ValueError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(report.model_dump_json(indent=2))
    if report.status == "error":
        for failure in report.failures:
            typer.secho(f"[ERROR] {failure}", err=True, fg=typer.colors.RED)
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


@eval_app.command("judge")
def eval_judge(
    run_id: Annotated[str, typer.Option(..., "--run-id")],
    questions: Annotated[
        Path,
        typer.Option(
            ...,
            "--questions",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    mode: Annotated[QueryMode, typer.Option("--mode")] = "hybrid",
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = 8,
    model: Annotated[str | None, typer.Option("--model")] = None,
    criteria: Annotated[list[str] | None, typer.Option("--criteria")] = None,
    out: Annotated[Path | None, typer.Option("--out")] = None,
) -> None:
    """Run LLM-as-a-judge scoring on answer outputs for a question set."""

    try:
        run_id = _validated_run_id(run_id)
        settings = load_settings()
        question_rows = _load_eval_questions(questions)
        artifact_dir = settings.artifact_root / run_id
        chunk_by_id = {chunk.chunk_id: chunk for chunk in load_chunks(artifact_dir)}
        resolved_model = (
            model.strip()
            if isinstance(model, str) and model.strip()
            else str(getattr(settings, "answer_model", "llama3"))
        )
        resolved_criteria_raw = criteria or ["correctness", "helpfulness", "groundedness"]
        allowed_criteria = {"correctness", "helpfulness", "groundedness", "coherence"}
        invalid_criteria = [
            criterion for criterion in resolved_criteria_raw if criterion not in allowed_criteria
        ]
        if invalid_criteria:
            raise AutoRAGError(
                "Invalid --criteria values: "
                f"{', '.join(sorted(set(invalid_criteria)))}. "
                "Use correctness, helpfulness, groundedness, or coherence."
            )
        resolved_criteria = [
            cast(JudgementCriteria, criterion) for criterion in resolved_criteria_raw
        ]
        output_path = out or (artifact_dir / "judge_results.jsonl")

        rows: list[dict[str, object]] = []
        for question in question_rows:
            payload = query_service(
                request=QueryRequest(
                    run_id=run_id,
                    question=question.question,
                    mode=mode,
                    top_k=top_k,
                ),
                settings=settings,
            )
            context = _context_for_hits(hits=payload.hits, chunk_by_id=chunk_by_id)
            judgements: list[dict[str, object]] = []
            for criterion in resolved_criteria:
                judgement = evaluate_with_llm_judge(
                    question=question.question,
                    answer=payload.answer.answer_text,
                    context=context,
                    criteria=criterion,
                    model=resolved_model,
                    ollama_base_url=settings.ollama_base_url,
                    timeout_seconds=settings.ollama_timeout_seconds,
                )
                judgements.append(judgement)

            total_score = 0.0
            for judgement in judgements:
                total_score += _coerce_float(judgement.get("score", 0.0))
            avg_score = total_score / float(len(judgements)) if judgements else 0.0
            rows.append(
                {
                    "run_id": run_id,
                    "question_id": question.question_id,
                    "question": question.question,
                    "mode": mode,
                    "top_k": top_k,
                    "answer": payload.answer.answer_text,
                    "judgements": judgements,
                    "avg_score": avg_score,
                }
            )

        write_jsonl_rows(output_path, rows)
        summary = {
            "run_id": run_id,
            "questions": len(rows),
            "mode": mode,
            "top_k": top_k,
            "model": resolved_model,
            "criteria": [str(item) for item in resolved_criteria],
            "output_path": str(output_path),
            "avg_score": (
                sum(_coerce_float(row.get("avg_score", 0.0)) for row in rows)
                / float(len(rows))
                if rows
                else 0.0
            ),
        }
        typer.echo(json.dumps(summary, indent=2))
    except (AutoRAGError, ValidationError, OSError, ValueError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


@eval_app.command("ab-test")
def eval_ab_test(
    run_id: Annotated[str, typer.Option(..., "--run-id")],
    questions: Annotated[
        Path,
        typer.Option(
            ...,
            "--questions",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    mode_a: Annotated[RetrievalMode, typer.Option("--mode-a")] = "vector",
    mode_b: Annotated[RetrievalMode, typer.Option("--mode-b")] = "hybrid",
    metric: Annotated[str, typer.Option("--metric")] = "recall_at_k",
    k: Annotated[int, typer.Option("--k", min=1)] = 5,
    n_runs: Annotated[int, typer.Option("--n-runs", min=1)] = 1,
    out: Annotated[Path | None, typer.Option("--out")] = None,
) -> None:
    """Run retrieval A/B testing and write markdown report."""

    try:
        run_id = _validated_run_id(run_id)
        settings = load_settings()
        question_rows = _load_eval_questions(questions)
        question_payloads = [row.model_dump(mode="json") for row in question_rows]

        def _run_retrieval(
            question_row: dict[str, object],
            mode: RetrievalMode,
            top_k: int,
        ) -> dict[str, float]:
            question_record = EvalQuestionRecord.model_validate(question_row)
            if mode == "vector":
                hits_for_metrics = run_vector_query_pipeline(
                    run_id=run_id,
                    question=question_record.question,
                    top_k=top_k,
                    settings=settings,
                )
            elif mode == "graph":
                hit_rows, _answer = run_graph_query_pipeline(
                    run_id=run_id,
                    question=question_record.question,
                    top_k=top_k,
                    settings=settings,
                )
                hits_for_metrics = [
                    RetrievalHitRecord.model_validate(row) for row in hit_rows
                ]
            else:
                hybrid_hits = run_hybrid_query_pipeline(
                    run_id=run_id,
                    question=question_record.question,
                    top_k=top_k,
                    settings=settings,
                )
                hits_for_metrics = [
                    RetrievalHitRecord(
                        question_id=hit.question_id,
                        rank=hit.rank,
                        score=hit.score,
                        chunk_id=hit.chunk_id,
                        doc_id=hit.doc_id,
                        page=hit.page,
                        section=hit.section,
                    )
                    for hit in hybrid_hits
                ]

            return {
                "recall_at_k": recall_at_k(
                    gold_citations=question_record.gold_citations,
                    hits=hits_for_metrics,
                    k=top_k,
                ),
                "ndcg_at_k": ndcg_at_k(
                    gold_citations=question_record.gold_citations,
                    hits=hits_for_metrics,
                    k=top_k,
                ),
            }

        result = run_ab_test(
            questions=question_payloads,
            run_retrieval_fn=_run_retrieval,
            mode_a=mode_a,
            mode_b=mode_b,
            metric=metric,
            k=k,
            n_runs=n_runs,
        )
        markdown = format_ab_test_report(result)

        reports_dir = Path("reports/experiments")
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = out or reports_dir / f"ab_test_{run_id}_{mode_a}_vs_{mode_b}.md"
        output_path.write_text(markdown, encoding="utf-8")
        json_path = output_path.with_suffix(".json")
        json_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

        typer.echo(
            json.dumps(
                {
                    "run_id": run_id,
                    "mode_a": mode_a,
                    "mode_b": mode_b,
                    "metric": metric,
                    "k": k,
                    "n_runs": n_runs,
                    "report_path": str(output_path),
                    "json_path": str(json_path),
                    "winner": result.winner,
                    "p_value": result.p_value,
                },
                indent=2,
            )
        )
    except (AutoRAGError, ValidationError, OSError, ValueError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


def main() -> None:
    """Console script entrypoint."""

    app()


if __name__ == "__main__":
    main()
