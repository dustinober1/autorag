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


def test_doctor_reports_missing_and_present_demo_prereqs(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runner = CliRunner()

    run_id = "m6_doctor"
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_file(input_dir / "fixture.pdf", "fixture text",)

    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / run_id
    _write_file(run_dir / "chunks.parquet", "ok")
    _write_file(run_dir / "embeddings.npy", "ok")

    reports_dir = tmp_path / "reports" / "milestones"
    matrix_reports_dir = tmp_path / "reports" / "experiments"
    _write_file(reports_dir / "m6_demo_report.md", "# report")

    monkeypatch.setenv("AUTORAG_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("AUTORAG_DEMO_RUN_ID", run_id)
    monkeypatch.setenv("AUTORAG_DEMO_INPUT_DIR", str(input_dir))
    monkeypatch.setenv("AUTORAG_DEMO_REPORTS_DIR", str(reports_dir))
    monkeypatch.setenv("AUTORAG_DEMO_MATRIX_REPORTS_DIR", str(matrix_reports_dir))

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1, result.output

    report = _parse_json_output(result.output)
    assert report.get("status") == "error"

    checks = report.get("checks")
    assert isinstance(checks, list)
    statuses = {
        row.get("status")
        for row in checks
        if isinstance(row, dict)
    }
    assert "present" in statuses
    assert "missing" in statuses

    missing_rows = [
        row
        for row in checks
        if isinstance(row, dict) and row.get("status") == "missing"
    ]
    assert missing_rows
    assert all(str(row.get("hint", "")).strip() for row in missing_rows)

    _write_file(run_dir / "embedding_meta.parquet", "ok")
    _write_file(run_dir / "kg.sqlite", "ok")
    _write_file(run_dir / "answers.jsonl", "{}\n")
    _write_file(run_dir / "demo_payload_samples.jsonl", "{}\n")
    _write_file(matrix_reports_dir / "matrix_results.csv", "run_id\nm6_doctor\n")
    _write_file(matrix_reports_dir / "matrix_results.json", '{"run_id":"m6_doctor","rows":[]}')
    _write_file(matrix_reports_dir / "leaderboard.md", "# leaderboard")

    healthy = runner.invoke(app, ["doctor"])
    assert healthy.exit_code == 0, healthy.output

    healthy_report = _parse_json_output(healthy.output)
    assert healthy_report.get("status") == "ok"
    assert healthy_report.get("missing") == 0
