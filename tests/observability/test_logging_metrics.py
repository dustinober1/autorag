from __future__ import annotations

import json
from pathlib import Path

from autokg_rag.config.settings import Settings
from autokg_rag.ingest.pipeline import run_smoke_pipeline


def _write_fixture_pdf(path: Path) -> None:
    path.write_text(
        "Project Scope\n"
        "Scope baseline is used to evaluate change requests and approve updates.\n",
        encoding="utf-8",
    )


def test_stage_logs_and_metric_events_emitted(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdfs"
    input_dir.mkdir(parents=True)
    _write_fixture_pdf(input_dir / "scope.pdf")

    settings = Settings(artifact_root=tmp_path / "artifacts")
    run_smoke_pipeline(
        input_dir=input_dir,
        question="What is project scope?",
        run_id="m1_observability",
        settings=settings,
    )

    artifact_dir = settings.artifact_root / "m1_observability"
    logs_path = artifact_dir / "logs.jsonl"
    metrics_path = artifact_dir / "metrics.jsonl"

    assert logs_path.exists()
    assert metrics_path.exists()

    logs = [json.loads(line) for line in logs_path.read_text(encoding="utf-8").splitlines() if line]
    assert logs
    assert all("run_id" in row and row["run_id"] == "m1_observability" for row in logs)
    assert all("stage" in row for row in logs)

    metrics = [
        json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert metrics
    assert any(str(row["metric_name"]).endswith(".count") for row in metrics)
