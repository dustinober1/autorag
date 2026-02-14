from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from pytest import MonkeyPatch
from typer.testing import CliRunner

from autokg_rag.cli import app


def _find_required_file(root: Path, filename: str) -> Path:
    for candidate in root.rglob(filename):
        return candidate
    raise AssertionError(f"Missing required artifact file: {filename}")


def test_eval_harness_writes_csv_json_and_markdown_report(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("AUTORAG_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.chdir(tmp_path)

    eval_help = runner.invoke(app, ["eval", "--help"])
    if eval_help.exit_code != 0:
        pytest.skip("Milestone 5 eval CLI commands are not available in this branch.")

    input_dir = tmp_path / "raw" / "pdfs"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "m5_fixture.pdf").write_text(
        (
            "Scope Control\n"
            "Scope control governs change approval and impact analysis.\f"
            "Risk Response\n"
            "Approved changes trigger risk reassessment and response updates.\n"
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
            "m4_fixture",
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
            "m4_fixture",
            "--embedding",
            "bge-small-en-v1.5",
        ],
    )
    assert index_result.exit_code == 0, index_result.output

    build_result = runner.invoke(app, ["build-kg", "--run-id", "m4_fixture"])
    assert build_result.exit_code == 0, build_result.output

    generate_result = runner.invoke(
        app,
        [
            "eval",
            "generate",
            "--run-id",
            "m5",
            "--input",
            str(artifact_root / "m4_fixture"),
            "--target-size",
            "200",
        ],
    )
    assert generate_result.exit_code == 0, generate_result.output
    assert generate_result.output.strip()

    config_path = tmp_path / "configs" / "experiments" / "matrix.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "factors": {
                    "chunking": ["heading_recursive"],
                    "embedding": ["bge-small-en-v1.5"],
                    "retrieval": ["hybrid"],
                },
                "chunking": ["heading_recursive"],
                "embeddings": ["bge-small-en-v1.5"],
                "retrieval_modes": ["hybrid"],
                "metrics": {"primary": "nDCG@10"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    matrix_result = runner.invoke(
        app,
        [
            "eval",
            "run-matrix",
            "--run-id",
            "m5",
            "--config",
            str(config_path),
        ],
    )
    assert matrix_result.exit_code == 0, matrix_result.output
    assert matrix_result.output.strip()

    report_result = runner.invoke(
        app,
        [
            "eval",
            "report",
            "--run-id",
            "m5",
        ],
    )
    assert report_result.exit_code == 0, report_result.output
    assert report_result.output.strip()

    generated_questions = sorted((artifact_root / "m5").glob("questions_*.jsonl"))
    assert generated_questions
    assert generated_questions[0].read_text(encoding="utf-8").strip()

    matrix_csv_path = _find_required_file(tmp_path, "matrix_results.csv")
    matrix_json_path = _find_required_file(tmp_path, "matrix_results.json")
    leaderboard_path = _find_required_file(tmp_path, "leaderboard.md")

    csv_lines = [line for line in matrix_csv_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(csv_lines) > 1
    with matrix_csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows
    assert any(row.get("run_id") == "m5" for row in csv_rows)
    assert any(row.get("retrieval") == "hybrid" for row in csv_rows)
    assert all(row.get("ndcg_at_10") not in ("", None) for row in csv_rows)

    matrix_json_payload = json.loads(matrix_json_path.read_text(encoding="utf-8"))
    assert isinstance(matrix_json_payload, dict)
    assert matrix_json_payload.get("run_id") == "m5"
    matrix_rows = matrix_json_payload.get("rows")
    assert isinstance(matrix_rows, list)
    assert matrix_rows
    assert all("exp_id" in row for row in matrix_rows if isinstance(row, dict))
    assert all("ndcg_at_10" in row for row in matrix_rows if isinstance(row, dict))

    leaderboard_text = leaderboard_path.read_text(encoding="utf-8")
    assert leaderboard_text.strip()
    assert "nDCG@10" in leaderboard_text
    assert "hybrid" in leaderboard_text

    per_query_files = [path for path in tmp_path.rglob("*.jsonl") if "per_query" in path.parts]
    assert per_query_files
    populated_files = [path for path in per_query_files if path.read_text(encoding="utf-8").strip()]
    assert populated_files

    first_per_query_row = json.loads(
        populated_files[0].read_text(encoding="utf-8").splitlines()[0]
    )
    assert first_per_query_row.get("question_id")
    assert first_per_query_row.get("answer")
    metrics = first_per_query_row.get("metrics")
    assert isinstance(metrics, dict)
    assert "ndcg@10" in metrics
