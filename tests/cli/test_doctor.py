from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pytest import MonkeyPatch
from typer.testing import CliRunner

from autokg_rag.cli import app


def _parse_json_output(output: str) -> dict[str, Any]:
    start = output.find("{")
    end = output.rfind("}")
    assert start >= 0 and end > start, output
    payload = json.loads(output[start : end + 1])
    assert isinstance(payload, dict)
    return payload


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _configure_demo_env(
    *,
    monkeypatch: MonkeyPatch,
    run_id: str,
    input_dir: Path,
    artifact_root: Path,
    reports_dir: Path,
    matrix_reports_dir: Path,
    embedding_provider: str = "local_hash",
    embedding_model: str = "bge-small-en-v1.5",
    reranker_enabled: str = "false",
    reranker_model: str = "llama3:8b",
    ollama_base_url: str = "http://localhost:11434",
) -> None:
    monkeypatch.setenv("AUTORAG_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("AUTORAG_DEMO_RUN_ID", run_id)
    monkeypatch.setenv("AUTORAG_DEMO_INPUT_DIR", str(input_dir))
    monkeypatch.setenv("AUTORAG_DEMO_REPORTS_DIR", str(reports_dir))
    monkeypatch.setenv("AUTORAG_DEMO_MATRIX_REPORTS_DIR", str(matrix_reports_dir))
    monkeypatch.setenv("AUTORAG_EMBEDDING_PROVIDER", embedding_provider)
    monkeypatch.setenv("AUTORAG_EMBEDDING_MODEL", embedding_model)
    monkeypatch.setenv("AUTORAG_RERANKER_ENABLED", reranker_enabled)
    monkeypatch.setenv("AUTORAG_RERANKER_MODEL", reranker_model)
    monkeypatch.setenv("AUTORAG_OLLAMA_BASE_URL", ollama_base_url)


def _seed_required_binary_artifacts(run_dir: Path) -> None:
    _write_file(run_dir / "chunks.parquet", "ok")
    _write_file(run_dir / "embeddings.npy", "ok")
    _write_file(run_dir / "embedding_meta.parquet", "ok")
    _write_file(run_dir / "kg.sqlite", "ok")


def test_doctor_reports_missing_artifacts_with_hints(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runner = CliRunner()

    run_id = "m6_doctor"
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_file(input_dir / "fixture.pdf", "fixture text")

    artifact_root = tmp_path / "artifacts"
    reports_dir = tmp_path / "reports" / "milestones"
    matrix_reports_dir = tmp_path / "reports" / "experiments"
    _configure_demo_env(
        monkeypatch=monkeypatch,
        run_id=run_id,
        input_dir=input_dir,
        artifact_root=artifact_root,
        reports_dir=reports_dir,
        matrix_reports_dir=matrix_reports_dir,
    )

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1, result.output

    report = _parse_json_output(result.output)
    assert report.get("status") == "error"
    assert int(report.get("invalid", -1)) == 0
    assert int(report.get("missing", 0)) > 0

    checks = report.get("checks", [])
    assert isinstance(checks, list)
    missing_rows = [
        row
        for row in checks
        if isinstance(row, dict) and row.get("status") == "missing"
    ]
    assert missing_rows
    assert all(str(row.get("hint", "")).strip() for row in missing_rows)
    assert "[MISSING]" in result.output


def test_doctor_reports_invalid_artifacts_with_hints(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runner = CliRunner()

    run_id = "m6_doctor_invalid"
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_file(input_dir / "fixture.pdf", "fixture text")

    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / run_id
    _seed_required_binary_artifacts(run_dir)
    _write_file(
        run_dir / "answers.jsonl",
        '{"question_id":"q1","answer_text":"ok","citations":[]}\n',
    )
    _write_file(
        run_dir / "demo_payload_samples.jsonl",
        '{"question":"Q","answer_record":{"citations":[]}}\n',
    )

    reports_dir = tmp_path / "reports" / "milestones"
    matrix_reports_dir = tmp_path / "reports" / "experiments"
    _write_file(reports_dir / "m6_demo_report.md", "   ")
    _write_file(matrix_reports_dir / "matrix_results.csv", "run_id,exp_id\n")
    _write_file(
        matrix_reports_dir / "matrix_results.json",
        '{"run_id":"m6_doctor_invalid","summary":{},"rows":[]}',
    )
    _write_file(matrix_reports_dir / "leaderboard.md", "\n")

    _configure_demo_env(
        monkeypatch=monkeypatch,
        run_id=run_id,
        input_dir=input_dir,
        artifact_root=artifact_root,
        reports_dir=reports_dir,
        matrix_reports_dir=matrix_reports_dir,
    )

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1, result.output

    report = _parse_json_output(result.output)
    assert report.get("status") == "error"
    assert int(report.get("missing", -1)) == 0
    assert int(report.get("invalid", 0)) > 0

    checks = report.get("checks", [])
    assert isinstance(checks, list)
    invalid_rows = [
        row
        for row in checks
        if isinstance(row, dict) and row.get("status") == "invalid"
    ]
    assert invalid_rows
    assert all(str(row.get("hint", "")).strip() for row in invalid_rows)

    invalid_names = {str(row.get("name")) for row in invalid_rows}
    assert "answers_jsonl" in invalid_names
    assert "demo_payload_samples_jsonl" in invalid_names
    assert "matrix_results_json" in invalid_names
    assert "matrix_results_csv" in invalid_names
    assert "m6_demo_report" in invalid_names
    assert "leaderboard_md" in invalid_names
    assert "[INVALID]" in result.output


def test_doctor_reports_valid_artifacts_and_exit_zero(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runner = CliRunner()

    run_id = "m6_doctor_valid"
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_file(input_dir / "fixture.pdf", "fixture text")

    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / run_id
    _seed_required_binary_artifacts(run_dir)
    _write_file(
        run_dir / "answers.jsonl",
        (
            '{"question_id":"q1","answer_text":"Scope baseline is controlled.",'
            '"citations":[{"chunk_id":"c1","doc_id":"d1","page":1,"section":"Scope"}]}\n'
        ),
    )
    _write_file(
        run_dir / "demo_payload_samples.jsonl",
        (
            '{"question":"What is scope control?",'
            '"answer_record":{"question_id":"q1","answer_text":"Scope control uses approvals.",'
            '"citations":[{"chunk_id":"c1","doc_id":"d1","page":1,"section":"Scope"}]}}\n'
        ),
    )

    reports_dir = tmp_path / "reports" / "milestones"
    matrix_reports_dir = tmp_path / "reports" / "experiments"
    _write_file(reports_dir / "m6_demo_report.md", "# Milestone report")
    _write_file(matrix_reports_dir / "matrix_results.csv", "run_id,exp_id\nm6_doctor_valid,exp_1\n")
    _write_file(
        matrix_reports_dir / "matrix_results.json",
        (
            '{"run_id":"m6_doctor_valid","summary":{"total_experiments":1},'
            '"rows":[{"run_id":"m6_doctor_valid","exp_id":"exp_1"}]}'
        ),
    )
    _write_file(matrix_reports_dir / "leaderboard.md", "# Leaderboard")

    _configure_demo_env(
        monkeypatch=monkeypatch,
        run_id=run_id,
        input_dir=input_dir,
        artifact_root=artifact_root,
        reports_dir=reports_dir,
        matrix_reports_dir=matrix_reports_dir,
    )

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output

    report = _parse_json_output(result.output)
    assert report.get("status") == "ok"
    assert int(report.get("missing", -1)) == 0
    assert int(report.get("invalid", -1)) == 0

    checks = report.get("checks", [])
    assert isinstance(checks, list)
    assert checks
    assert all(
        isinstance(row, dict) and row.get("status") == "valid"
        for row in checks
    )
    check_names = {
        str(row.get("name"))
        for row in checks
        if isinstance(row, dict)
    }
    assert not any(name.startswith("ollama_") for name in check_names)


def test_doctor_runs_optional_ollama_checks_with_actionable_hints(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runner = CliRunner()

    run_id = "m6_doctor_ollama"
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_file(input_dir / "fixture.pdf", "fixture text")

    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / run_id
    _seed_required_binary_artifacts(run_dir)
    _write_file(
        run_dir / "answers.jsonl",
        (
            '{"question_id":"q1","answer_text":"Scope baseline is controlled.",'
            '"citations":[{"chunk_id":"c1","doc_id":"d1","page":1,"section":"Scope"}]}\n'
        ),
    )
    _write_file(
        run_dir / "demo_payload_samples.jsonl",
        (
            '{"question":"What is scope control?",'
            '"answer_record":{"question_id":"q1","answer_text":"Scope control uses approvals.",'
            '"citations":[{"chunk_id":"c1","doc_id":"d1","page":1,"section":"Scope"}]}}\n'
        ),
    )

    reports_dir = tmp_path / "reports" / "milestones"
    matrix_reports_dir = tmp_path / "reports" / "experiments"
    _write_file(reports_dir / "m6_demo_report.md", "# Milestone report")
    _write_file(
        matrix_reports_dir / "matrix_results.csv",
        "run_id,exp_id\nm6_doctor_ollama,exp_1\n",
    )
    _write_file(
        matrix_reports_dir / "matrix_results.json",
        (
            '{"run_id":"m6_doctor_ollama","summary":{"total_experiments":1},'
            '"rows":[{"run_id":"m6_doctor_ollama","exp_id":"exp_1"}]}'
        ),
    )
    _write_file(matrix_reports_dir / "leaderboard.md", "# Leaderboard")

    _configure_demo_env(
        monkeypatch=monkeypatch,
        run_id=run_id,
        input_dir=input_dir,
        artifact_root=artifact_root,
        reports_dir=reports_dir,
        matrix_reports_dir=matrix_reports_dir,
        embedding_provider="ollama",
        embedding_model="embeddinggemma:300m",
        reranker_enabled="true",
        reranker_model="llama3:8b",
        ollama_base_url="http://127.0.0.1:9",
    )

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1, result.output

    report = _parse_json_output(result.output)
    assert report.get("status") == "error"

    checks = report.get("checks", [])
    assert isinstance(checks, list)
    checks_by_name = {
        str(row.get("name")): row
        for row in checks
        if isinstance(row, dict)
    }
    for expected_name in (
        "ollama_reachability",
        "ollama_model_embedding",
        "ollama_model_reranker",
        "ollama_embeddings_endpoint",
        "ollama_generate_endpoint",
    ):
        row = checks_by_name.get(expected_name)
        assert isinstance(row, dict), expected_name
        assert row.get("status") in {"missing", "invalid"}, expected_name
        hint = str(row.get("hint", "")).lower()
        assert "ollama serve" in hint or "ollama pull" in hint
