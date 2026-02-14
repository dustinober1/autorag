from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_demo_smoke_target_produces_minimum_m6_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    input_dir = tmp_path / "raw" / "pdfs"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "m6_smoke_fixture.pdf").write_text(
        (
            "Scope Control\n"
            "Scope control balances scope changes against risk acceptance.\f"
            "Risk Response\n"
            "Mitigation and acceptance are selected using trigger thresholds.\n"
        ),
        encoding="utf-8",
    )

    artifact_root = tmp_path / "artifacts"
    reports_root = tmp_path / "reports"

    env = os.environ.copy()
    env["AUTORAG_ARTIFACT_ROOT"] = str(artifact_root)
    env["AUTORAG_DEMO_RUN_ID"] = "m6_smoke"
    env["AUTORAG_DEMO_INPUT_DIR"] = str(input_dir)
    env["AUTORAG_DEMO_REPORTS_DIR"] = str(reports_root / "milestones")
    env["AUTORAG_DEMO_MATRIX_REPORTS_DIR"] = str(reports_root / "experiments")
    env["AUTORAG_DEMO_MODE"] = "vector"
    env["AUTORAG_DEMO_TOP_K"] = "4"

    result = subprocess.run(
        ["make", "demo-smoke"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    run_dir = artifact_root / "m6_smoke"
    required_paths = [
        run_dir / "chunks.parquet",
        run_dir / "embeddings.npy",
        run_dir / "kg.sqlite",
        run_dir / "answers.jsonl",
        run_dir / "demo_payload_samples.jsonl",
        reports_root / "milestones" / "m6_demo_report.md",
        reports_root / "experiments" / "matrix_results.csv",
        reports_root / "experiments" / "matrix_results.json",
    ]
    for path in required_paths:
        assert path.exists(), str(path)

    payload_rows = [
        json.loads(line)
        for line in (run_dir / "demo_payload_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert payload_rows
    first_row = payload_rows[0]
    assert first_row.get("question")
    answer_record = first_row.get("answer_record")
    assert isinstance(answer_record, dict)
    assert answer_record.get("answer_text")
