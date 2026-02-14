from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

from autokg_rag.app_api.service import query_service
from autokg_rag.cli import app
from autokg_rag.config import load_settings
from autokg_rag.schemas.api import QueryRequest


def test_query_service_returns_answerrecord_with_citations(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("AUTORAG_ARTIFACT_ROOT", str(artifact_root))

    input_dir = tmp_path / "raw" / "pdfs"
    input_dir.mkdir(parents=True)
    (input_dir / "api_fixture.pdf").write_text(
        (
            "Scope Control\n"
            "Scope control governs approval and mitigation planning.\f"
            "Risk Response\n"
            "Mitigation options are compared against acceptance triggers.\n"
        ),
        encoding="utf-8",
    )

    ingest_result = runner.invoke(
        app,
        [
            "ingest",
            "--input",
            str(input_dir),
            "--run-id",
            "m6",
            "--chunking",
            "heading_recursive",
        ],
    )
    assert ingest_result.exit_code == 0, ingest_result.output

    index_result = runner.invoke(
        app,
        [
            "index-vector",
            "--run-id",
            "m6",
            "--embedding",
            "bge-small-en-v1.5",
        ],
    )
    assert index_result.exit_code == 0, index_result.output

    build_result = runner.invoke(app, ["build-kg", "--run-id", "m6"])
    assert build_result.exit_code == 0, build_result.output

    payload = query_service(
        request=QueryRequest(
            run_id="m6",
            question="Compare mitigation and acceptance strategies.",
            mode="hybrid",
            top_k=5,
        ),
        settings=load_settings(),
    )

    assert payload.answer.answer_text.strip()
    assert payload.answer.citations
    assert payload.citation_trace
    assert payload.hits
    assert payload.answer.question_id == payload.hits[0].question_id
    assert all(citation.chunk_id for citation in payload.answer.citations)

    sample_path = artifact_root / "m6" / "demo_payload_samples.jsonl"
    assert sample_path.exists()
    assert sample_path.read_text(encoding="utf-8").strip()
