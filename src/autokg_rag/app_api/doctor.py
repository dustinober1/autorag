"""Doctor checks for local Milestone 6 demo prerequisites and artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

CheckStatus = Literal["valid", "missing", "invalid"]
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
)

_ANSWERS_HINT = (
    "Run `make demo-build` to regenerate `answers.jsonl`; rows must include "
    "non-empty `answer_text` and `citations`."
)
_DEMO_PAYLOAD_HINT = (
    "Run `make demo-build` to regenerate `demo_payload_samples.jsonl`; rows must include "
    "`question` and `answer_record.answer_text`."
)
_MATRIX_JSON_HINT = (
    "Run `make demo-build` to regenerate `matrix_results.json`; expected keys are "
    "`run_id`, `summary`, and non-empty `rows`."
)
_MATRIX_CSV_HINT = (
    "Run `make demo-build` to regenerate `matrix_results.csv`; file must have a header and "
    "at least one data row."
)
_M6_REPORT_HINT = "Run `make demo-build` to regenerate milestone report output."
_LEADERBOARD_HINT = "Run `make demo-build` to regenerate leaderboard report."


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
    valid: int
    invalid: int
    missing: int
    present: int
    checks: list[DoctorCheck]


def _valid_check(*, name: str, path: Path, message: str) -> DoctorCheck:
    return DoctorCheck(
        name=name,
        status="valid",
        path=str(path),
        message=message,
    )


def _missing_check(*, name: str, path: Path, message: str, hint: str) -> DoctorCheck:
    return DoctorCheck(
        name=name,
        status="missing",
        path=str(path),
        message=message,
        hint=hint,
    )


def _invalid_check(*, name: str, path: Path, message: str, hint: str) -> DoctorCheck:
    return DoctorCheck(
        name=name,
        status="invalid",
        path=str(path),
        message=message,
        hint=hint,
    )


def _directory_check(*, name: str, path: Path, hint: str) -> DoctorCheck:
    if path.is_dir():
        return _valid_check(name=name, path=path, message="Found required directory.")
    if path.exists():
        return _invalid_check(
            name=name,
            path=path,
            message="Path exists but is not a directory.",
            hint=hint,
        )
    return _missing_check(
        name=name,
        path=path,
        message="Missing required directory.",
        hint=hint,
    )


def _required_file_check(*, name: str, path: Path, hint: str) -> DoctorCheck:
    if path.is_file():
        return _valid_check(name=name, path=path, message=f"Found required file: {path.name}.")
    if path.exists():
        return _invalid_check(
            name=name,
            path=path,
            message=f"Path exists but is not a file: {path.name}.",
            hint=hint,
        )
    return _missing_check(
        name=name,
        path=path,
        message=f"Missing required file: {path.name}.",
        hint=hint,
    )


def _input_pdf_check(input_dir: Path) -> DoctorCheck:
    if not input_dir.exists():
        return _missing_check(
            name="demo_input_pdfs",
            path=input_dir,
            message="Demo input directory is missing, cannot scan for PDF files.",
            hint="Set `AUTORAG_DEMO_INPUT_DIR` to a directory containing at least one `.pdf`.",
        )
    if not input_dir.is_dir():
        return _invalid_check(
            name="demo_input_pdfs",
            path=input_dir,
            message="Demo input path exists but is not a directory.",
            hint="Set `AUTORAG_DEMO_INPUT_DIR` to a directory containing at least one `.pdf`.",
        )

    pdf_count = sum(
        1
        for candidate in input_dir.iterdir()
        if candidate.is_file() and candidate.suffix.lower() == ".pdf"
    )
    if pdf_count > 0:
        return _valid_check(
            name="demo_input_pdfs",
            path=input_dir,
            message=f"Found {pdf_count} PDF file(s) for demo ingestion.",
        )

    return _invalid_check(
        name="demo_input_pdfs",
        path=input_dir,
        message="No PDF files found in demo input directory.",
        hint="Add at least one `.pdf` file or point `AUTORAG_DEMO_INPUT_DIR` to fixture PDFs.",
    )


def _read_jsonl_rows(
    *,
    name: str,
    path: Path,
    hint: str,
) -> tuple[list[dict[str, Any]] | None, DoctorCheck | None]:
    file_check = _required_file_check(name=name, path=path, hint=hint)
    if file_check.status != "valid":
        return None, file_check

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return None, _invalid_check(
            name=name,
            path=path,
            message=f"Unable to read text file: {exc}.",
            hint=hint,
        )

    rows: list[dict[str, Any]] = []
    parsed_line_count = 0
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parsed_line_count += 1
        try:
            raw_row = json.loads(line)
        except json.JSONDecodeError as exc:
            return None, _invalid_check(
                name=name,
                path=path,
                message=f"Invalid JSON on line {line_no}: {exc.msg}.",
                hint=hint,
            )
        if not isinstance(raw_row, dict):
            return None, _invalid_check(
                name=name,
                path=path,
                message=f"Row {line_no} is not a JSON object.",
                hint=hint,
            )
        rows.append(raw_row)

    if parsed_line_count == 0:
        return None, _invalid_check(
            name=name,
            path=path,
            message="File is empty.",
            hint=hint,
        )
    return rows, None


def _is_non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_answers_jsonl(path: Path) -> DoctorCheck:
    name = "answers_jsonl"
    rows, error_check = _read_jsonl_rows(name=name, path=path, hint=_ANSWERS_HINT)
    if error_check is not None:
        return error_check
    assert rows is not None

    for index, row in enumerate(rows, start=1):
        if not _is_non_empty_text(row.get("answer_text")):
            return _invalid_check(
                name=name,
                path=path,
                message=f"Row {index} is missing non-empty `answer_text`.",
                hint=_ANSWERS_HINT,
            )
        citations = row.get("citations")
        if not isinstance(citations, list) or len(citations) == 0:
            return _invalid_check(
                name=name,
                path=path,
                message=f"Row {index} is missing non-empty `citations`.",
                hint=_ANSWERS_HINT,
            )

    return _valid_check(
        name=name,
        path=path,
        message=f"Validated {len(rows)} answer row(s) with required fields.",
    )


def _validate_demo_payload_samples_jsonl(path: Path) -> DoctorCheck:
    name = "demo_payload_samples_jsonl"
    rows, error_check = _read_jsonl_rows(name=name, path=path, hint=_DEMO_PAYLOAD_HINT)
    if error_check is not None:
        return error_check
    assert rows is not None

    for index, row in enumerate(rows, start=1):
        if not _is_non_empty_text(row.get("question")):
            return _invalid_check(
                name=name,
                path=path,
                message=f"Row {index} is missing non-empty `question`.",
                hint=_DEMO_PAYLOAD_HINT,
            )
        answer_record = row.get("answer_record")
        if not isinstance(answer_record, dict):
            return _invalid_check(
                name=name,
                path=path,
                message=f"Row {index} is missing object `answer_record`.",
                hint=_DEMO_PAYLOAD_HINT,
            )
        if not _is_non_empty_text(answer_record.get("answer_text")):
            return _invalid_check(
                name=name,
                path=path,
                message=f"Row {index} is missing non-empty `answer_record.answer_text`.",
                hint=_DEMO_PAYLOAD_HINT,
            )

    return _valid_check(
        name=name,
        path=path,
        message=f"Validated {len(rows)} demo payload sample row(s).",
    )


def _validate_matrix_results_json(path: Path) -> DoctorCheck:
    name = "matrix_results_json"
    file_check = _required_file_check(name=name, path=path, hint=_MATRIX_JSON_HINT)
    if file_check.status != "valid":
        return file_check

    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return _invalid_check(
            name=name,
            path=path,
            message=f"Unable to read JSON file: {exc}.",
            hint=_MATRIX_JSON_HINT,
        )
    if not raw_text.strip():
        return _invalid_check(
            name=name,
            path=path,
            message="JSON file is empty.",
            hint=_MATRIX_JSON_HINT,
        )

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return _invalid_check(
            name=name,
            path=path,
            message=f"Invalid JSON object: {exc.msg}.",
            hint=_MATRIX_JSON_HINT,
        )
    if not isinstance(payload, dict):
        return _invalid_check(
            name=name,
            path=path,
            message="Expected top-level JSON object.",
            hint=_MATRIX_JSON_HINT,
        )

    expected_keys = ("run_id", "summary", "rows")
    missing_keys = [key for key in expected_keys if key not in payload]
    if missing_keys:
        return _invalid_check(
            name=name,
            path=path,
            message=f"Missing top-level key(s): {', '.join(missing_keys)}.",
            hint=_MATRIX_JSON_HINT,
        )
    if not isinstance(payload.get("summary"), dict):
        return _invalid_check(
            name=name,
            path=path,
            message="Top-level `summary` must be a JSON object.",
            hint=_MATRIX_JSON_HINT,
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return _invalid_check(
            name=name,
            path=path,
            message="Top-level `rows` must be a JSON array.",
            hint=_MATRIX_JSON_HINT,
        )
    if len(rows) == 0:
        return _invalid_check(
            name=name,
            path=path,
            message="Top-level `rows` array is empty.",
            hint=_MATRIX_JSON_HINT,
        )
    if not all(isinstance(row, dict) for row in rows):
        return _invalid_check(
            name=name,
            path=path,
            message="Top-level `rows` contains non-object entries.",
            hint=_MATRIX_JSON_HINT,
        )

    return _valid_check(
        name=name,
        path=path,
        message=f"Validated matrix JSON with {len(rows)} row(s).",
    )


def _validate_matrix_results_csv(path: Path) -> DoctorCheck:
    name = "matrix_results_csv"
    file_check = _required_file_check(name=name, path=path, hint=_MATRIX_CSV_HINT)
    if file_check.status != "valid":
        return file_check

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames
            if not header or any(not field or not field.strip() for field in header):
                return _invalid_check(
                    name=name,
                    path=path,
                    message="CSV header is missing or empty.",
                    hint=_MATRIX_CSV_HINT,
                )
            data_rows = 0
            for row in reader:
                if any((value or "").strip() for value in row.values()):
                    data_rows += 1
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return _invalid_check(
            name=name,
            path=path,
            message=f"Unable to parse CSV file: {exc}.",
            hint=_MATRIX_CSV_HINT,
        )

    if data_rows == 0:
        return _invalid_check(
            name=name,
            path=path,
            message="CSV has no data rows.",
            hint=_MATRIX_CSV_HINT,
        )

    return _valid_check(
        name=name,
        path=path,
        message=f"Validated matrix CSV with {data_rows} row(s).",
    )


def _validate_non_empty_text_file(*, name: str, path: Path, hint: str) -> DoctorCheck:
    file_check = _required_file_check(name=name, path=path, hint=hint)
    if file_check.status != "valid":
        return file_check

    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return _invalid_check(
            name=name,
            path=path,
            message=f"Unable to read text file: {exc}.",
            hint=hint,
        )
    if not content.strip():
        return _invalid_check(
            name=name,
            path=path,
            message=f"File is empty: {path.name}.",
            hint=hint,
        )
    return _valid_check(
        name=name,
        path=path,
        message=f"Validated non-empty text file: {path.name}.",
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
        checks.append(_required_file_check(name=name, path=run_artifact_dir / filename, hint=hint))

    checks.append(_validate_answers_jsonl(run_artifact_dir / "answers.jsonl"))
    checks.append(
        _validate_demo_payload_samples_jsonl(run_artifact_dir / "demo_payload_samples.jsonl")
    )
    checks.append(
        _validate_non_empty_text_file(
            name="m6_demo_report",
            path=reports_dir / "m6_demo_report.md",
            hint=_M6_REPORT_HINT,
        )
    )
    checks.append(_validate_matrix_results_csv(matrix_reports_dir / "matrix_results.csv"))
    checks.append(_validate_matrix_results_json(matrix_reports_dir / "matrix_results.json"))
    checks.append(
        _validate_non_empty_text_file(
            name="leaderboard_md",
            path=matrix_reports_dir / "leaderboard.md",
            hint=_LEADERBOARD_HINT,
        )
    )

    valid = sum(1 for check in checks if check.status == "valid")
    invalid = sum(1 for check in checks if check.status == "invalid")
    missing = sum(1 for check in checks if check.status == "missing")
    status: DoctorStatus = "ok" if invalid == 0 and missing == 0 else "error"

    return DoctorReport(
        run_id=run_id,
        status=status,
        valid=valid,
        invalid=invalid,
        missing=missing,
        present=valid,
        checks=checks,
    )
