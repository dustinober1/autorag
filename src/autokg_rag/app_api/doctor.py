"""Doctor checks for local Milestone 6 demo prerequisites and artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

CheckStatus = Literal["present", "missing"]
DoctorStatus = Literal["ok", "error"]

_REQUIRED_RUN_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    (
        "chunks_parquet",
        "chunks.parquet",
        "Run `make demo-build` to regenerate ingest artifacts.",
    ),
    (
        "embeddings_npy",
        "embeddings.npy",
        "Run `make demo-build` to regenerate vector embeddings.",
    ),
    (
        "embedding_meta_parquet",
        "embedding_meta.parquet",
        "Run `make demo-build` to regenerate embedding metadata.",
    ),
    (
        "kg_sqlite",
        "kg.sqlite",
        "Run `make demo-build` to rebuild knowledge graph storage.",
    ),
    (
        "answers_jsonl",
        "answers.jsonl",
        "Run `make demo-build` to write grounded answer artifacts.",
    ),
    (
        "demo_payload_samples_jsonl",
        "demo_payload_samples.jsonl",
        "Run `make demo-build` to regenerate demo payload samples.",
    ),
)

_REQUIRED_REPORT_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    (
        "m6_demo_report",
        "m6_demo_report.md",
        "Run `make demo-build` to regenerate milestone report output.",
    ),
    (
        "matrix_results_csv",
        "matrix_results.csv",
        "Run `make demo-build` to regenerate experiment matrix CSV.",
    ),
    (
        "matrix_results_json",
        "matrix_results.json",
        "Run `make demo-build` to regenerate experiment matrix JSON.",
    ),
    (
        "leaderboard_md",
        "leaderboard.md",
        "Run `make demo-build` to regenerate leaderboard report.",
    ),
)


class DoctorCheck(BaseModel):
    """One doctor check entry."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: CheckStatus
    path: str
    message: str
    hint: str | None = None


class DoctorReport(BaseModel):
    """Aggregated doctor report."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: DoctorStatus
    present: int
    missing: int
    checks: list[DoctorCheck]


def _file_check(*, name: str, path: Path, hint: str) -> DoctorCheck:
    if path.is_file():
        return DoctorCheck(
            name=name,
            status="present",
            path=str(path),
            message=f"Found required file: {path.name}.",
        )
    return DoctorCheck(
        name=name,
        status="missing",
        path=str(path),
        message=f"Missing required file: {path.name}.",
        hint=hint,
    )


def _directory_check(*, name: str, path: Path, hint: str) -> DoctorCheck:
    if path.is_dir():
        return DoctorCheck(
            name=name,
            status="present",
            path=str(path),
            message="Found required directory.",
        )
    return DoctorCheck(
        name=name,
        status="missing",
        path=str(path),
        message="Missing required directory.",
        hint=hint,
    )


def _input_pdf_check(input_dir: Path) -> DoctorCheck:
    if not input_dir.is_dir():
        return DoctorCheck(
            name="demo_input_pdfs",
            status="missing",
            path=str(input_dir),
            message="Demo input directory is missing, cannot scan for PDF files.",
            hint="Set `AUTORAG_DEMO_INPUT_DIR` to a directory containing at least one `.pdf`.",
        )

    pdf_count = sum(
        1
        for candidate in input_dir.iterdir()
        if candidate.is_file() and candidate.suffix.lower() == ".pdf"
    )
    if pdf_count > 0:
        return DoctorCheck(
            name="demo_input_pdfs",
            status="present",
            path=str(input_dir),
            message=f"Found {pdf_count} PDF file(s) for demo ingestion.",
        )

    return DoctorCheck(
        name="demo_input_pdfs",
        status="missing",
        path=str(input_dir),
        message="No PDF files found in demo input directory.",
        hint="Add at least one `.pdf` file or point `AUTORAG_DEMO_INPUT_DIR` to fixture PDFs.",
    )


def run_demo_doctor(
    *,
    run_id: str,
    input_dir: Path,
    artifact_root: Path,
    reports_dir: Path,
    matrix_reports_dir: Path,
) -> DoctorReport:
    """Validate local prerequisites and outputs for M6 demo workflows."""

    checks: list[DoctorCheck] = []
    checks.append(
        _directory_check(
            name="demo_input_dir",
            path=input_dir,
            hint="Set `AUTORAG_DEMO_INPUT_DIR` to an existing directory.",
        )
    )
    checks.append(_input_pdf_check(input_dir))

    run_artifact_dir = artifact_root / run_id
    checks.append(
        _directory_check(
            name="run_artifact_dir",
            path=run_artifact_dir,
            hint=(
                f"Run `make demo-build` with `AUTORAG_DEMO_RUN_ID={run_id}` "
                "to create this artifact directory."
            ),
        )
    )

    for name, filename, hint in _REQUIRED_RUN_ARTIFACTS:
        checks.append(_file_check(name=name, path=run_artifact_dir / filename, hint=hint))

    for name, filename, hint in _REQUIRED_REPORT_ARTIFACTS:
        base_dir = reports_dir if filename == "m6_demo_report.md" else matrix_reports_dir
        checks.append(_file_check(name=name, path=base_dir / filename, hint=hint))

    missing = sum(1 for check in checks if check.status == "missing")
    present = len(checks) - missing
    status: DoctorStatus = "ok" if missing == 0 else "error"

    return DoctorReport(
        run_id=run_id,
        status=status,
        present=present,
        missing=missing,
        checks=checks,
    )
