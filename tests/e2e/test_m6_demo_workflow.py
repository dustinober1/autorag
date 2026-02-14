from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_demo_build_generates_required_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    input_dir = tmp_path / "raw" / "pdfs"
    input_dir.mkdir(parents=True)
    (input_dir / "m6_fixture.pdf").write_text(
        (
            "Scope Control\n"
            "Scope control requires approval and mitigation tradeoff analysis.\f"
            "Risk Response\n"
            "Acceptance and mitigation are compared by trigger thresholds.\n"
        ),
        encoding="utf-8",
    )

    artifact_root = tmp_path / "artifacts"
    reports_root = tmp_path / "reports"

    env = os.environ.copy()
    env["AUTORAG_ARTIFACT_ROOT"] = str(artifact_root)
    env["AUTORAG_DEMO_RUN_ID"] = "m6"
    env["AUTORAG_DEMO_INPUT_DIR"] = str(input_dir)
    env["AUTORAG_DEMO_REPORTS_DIR"] = str(reports_root / "milestones")
    env["AUTORAG_DEMO_MATRIX_REPORTS_DIR"] = str(reports_root / "experiments")

    result = subprocess.run(
        ["make", "demo-build"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    payload_path = artifact_root / "m6" / "demo_payload_samples.jsonl"
    answers_path = artifact_root / "m6" / "answers.jsonl"
    report_path = reports_root / "milestones" / "m6_demo_report.md"

    assert payload_path.exists()
    assert answers_path.exists()
    assert report_path.exists()

    payload_lines = [line for line in payload_path.read_text(encoding="utf-8").splitlines() if line]
    assert payload_lines
    row = json.loads(payload_lines[0])
    assert row.get("question")
    answer_record = row.get("answer_record")
    assert isinstance(answer_record, dict)
    assert answer_record.get("answer_text")
    assert answer_record.get("citations")

    assert report_path.read_text(encoding="utf-8").strip()
