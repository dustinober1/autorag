"""Doctor checks for local Milestone 6 demo prerequisites and artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal
from urllib import error as urllib_error
from urllib import request as urllib_request

from pydantic import BaseModel, ConfigDict

from autokg_rag.config import Settings

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
_DEFAULT_EMBEDDING_MODEL = "bge-small-en-v1.5"
_DEFAULT_RERANKER_MODEL = "llama3:8b"
_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


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


def _valid_check(*, name: str, path: Path | str, message: str) -> DoctorCheck:
    return DoctorCheck(
        name=name,
        status="valid",
        path=str(path),
        message=message,
    )


def _missing_check(*, name: str, path: Path | str, message: str, hint: str) -> DoctorCheck:
    return DoctorCheck(
        name=name,
        status="missing",
        path=str(path),
        message=message,
        hint=hint,
    )


def _invalid_check(*, name: str, path: Path | str, message: str, hint: str) -> DoctorCheck:
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


def _ollama_request_json(
    *,
    base_url: str,
    endpoint: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib_request.Request(url=url, data=data, method=method, headers=headers)
    with urllib_request.urlopen(req, timeout=3.0) as response:
        status = int(getattr(response, "status", 200))
        body = response.read().decode("utf-8")
    decoded = json.loads(body) if body.strip() else {}
    return status, decoded


def _normalize_model_name(raw: str) -> str:
    text = raw.strip()
    return text if text else "unknown"


def _model_variants(model_name: str) -> set[str]:
    clean = _normalize_model_name(model_name)
    variants = {clean}
    if ":" in clean:
        variants.add(clean.split(":", 1)[0])
    else:
        variants.add(f"{clean}:latest")
    return variants


def _available_models_from_tags(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return set()

    available: set[str] = set()
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            value = item.get(key)
            if not isinstance(value, str):
                continue
            normalized = value.strip()
            if not normalized:
                continue
            available.update(_model_variants(normalized))
    return available


def _ollama_reachability_check(base_url: str) -> tuple[DoctorCheck, set[str] | None]:
    name = "ollama_reachability"
    path = f"{base_url.rstrip('/')}/api/tags"
    hint = (
        "Run `ollama serve` and confirm `AUTORAG_OLLAMA_BASE_URL` points to the active API "
        "(default `http://localhost:11434`)."
    )
    try:
        status, payload = _ollama_request_json(base_url=base_url, endpoint="/api/tags")
    except urllib_error.HTTPError as exc:
        return (
            _invalid_check(
                name=name,
                path=path,
                message=f"Ollama tags endpoint returned HTTP {exc.code}.",
                hint=hint,
            ),
            None,
        )
    except (urllib_error.URLError, TimeoutError, OSError, ValueError) as exc:
        return (
            _missing_check(
                name=name,
                path=path,
                message=f"Unable to reach Ollama API: {exc}.",
                hint=hint,
            ),
            None,
        )

    models = _available_models_from_tags(payload)
    return (
        _valid_check(
            name=name,
            path=path,
            message=f"Ollama API reachable (HTTP {status}); discovered {len(models)} model tag(s).",
        ),
        models,
    )


def _ollama_model_check(
    *,
    model_role: str,
    model_name: str,
    available_models: set[str] | None,
) -> DoctorCheck:
    normalized = _normalize_model_name(model_name)
    check_name = f"ollama_model_{model_role}"
    path = f"ollama://model/{normalized}"
    hint = f"Run `ollama pull {normalized}` (after `ollama serve`) and retry."
    if available_models is None:
        return _missing_check(
            name=check_name,
            path=path,
            message="Unable to verify model availability because Ollama API is unreachable.",
            hint=hint,
        )

    if _model_variants(normalized) & available_models:
        return _valid_check(
            name=check_name,
            path=path,
            message=f"Ollama model is available for {model_role}: {normalized}.",
        )

    return _missing_check(
        name=check_name,
        path=path,
        message=f"Ollama model is not available for {model_role}: {normalized}.",
        hint=hint,
    )


def _embedding_vector_length(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None

    single = payload.get("embedding")
    if isinstance(single, list) and single:
        return len(single)

    multi = payload.get("embeddings")
    if isinstance(multi, list) and multi:
        first = multi[0]
        if isinstance(first, list) and first:
            return len(first)
        if isinstance(first, (float, int)):
            return len(multi)
    return None


def _ollama_embeddings_health_check(
    *,
    base_url: str,
    model_name: str,
    ollama_reachable: bool,
) -> DoctorCheck:
    normalized = _normalize_model_name(model_name)
    health_url = f"{base_url.rstrip('/')}/api/embeddings"
    hint = (
        f"Run `ollama serve` and `ollama pull {normalized}`, then retry the doctor check."
    )
    if not ollama_reachable:
        return _missing_check(
            name="ollama_embeddings_endpoint",
            path=health_url,
            message="Unable to validate embeddings endpoint because Ollama API is unreachable.",
            hint=hint,
        )

    attempts = (
        ("/api/embeddings", {"model": normalized, "prompt": "doctor healthcheck"}),
        ("/api/embed", {"model": normalized, "input": "doctor healthcheck"}),
    )
    errors: list[str] = []
    for endpoint, payload in attempts:
        endpoint_url = f"{base_url.rstrip('/')}{endpoint}"
        try:
            status, response_payload = _ollama_request_json(
                base_url=base_url,
                endpoint=endpoint,
                method="POST",
                payload=payload,
            )
        except urllib_error.HTTPError as exc:
            errors.append(f"{endpoint}: HTTP {exc.code}")
            continue
        except (urllib_error.URLError, TimeoutError, OSError, ValueError) as exc:
            errors.append(f"{endpoint}: {exc}")
            continue

        embedding_len = _embedding_vector_length(response_payload)
        if embedding_len is not None and embedding_len > 0:
            return _valid_check(
                name="ollama_embeddings_endpoint",
                path=endpoint_url,
                message=(
                    f"Embeddings endpoint healthy via {endpoint} (HTTP {status}); "
                    f"vector length {embedding_len}."
                ),
            )
        errors.append(f"{endpoint}: response missing embedding vector")

    summary = "; ".join(errors) if errors else "unknown embeddings endpoint failure"
    return _invalid_check(
        name="ollama_embeddings_endpoint",
        path=health_url,
        message=(
            "Ollama embeddings endpoint health check failed for "
            f"model '{normalized}': {summary}."
        ),
        hint=hint,
    )


def _ollama_generate_health_check(
    *,
    base_url: str,
    model_name: str,
    ollama_reachable: bool,
) -> DoctorCheck:
    normalized = _normalize_model_name(model_name)
    endpoint_url = f"{base_url.rstrip('/')}/api/generate"
    hint = (
        f"Run `ollama serve` and `ollama pull {normalized}`, then retry the doctor check."
    )
    if not ollama_reachable:
        return _missing_check(
            name="ollama_generate_endpoint",
            path=endpoint_url,
            message="Unable to validate generate endpoint because Ollama API is unreachable.",
            hint=hint,
        )

    try:
        status, response_payload = _ollama_request_json(
            base_url=base_url,
            endpoint="/api/generate",
            method="POST",
            payload={
                "model": normalized,
                "prompt": "doctor healthcheck",
                "stream": False,
            },
        )
    except urllib_error.HTTPError as exc:
        return _invalid_check(
            name="ollama_generate_endpoint",
            path=endpoint_url,
            message=f"Ollama generate endpoint returned HTTP {exc.code}.",
            hint=hint,
        )
    except (urllib_error.URLError, TimeoutError, OSError, ValueError) as exc:
        return _invalid_check(
            name="ollama_generate_endpoint",
            path=endpoint_url,
            message=f"Ollama generate endpoint request failed: {exc}.",
            hint=hint,
        )

    if not isinstance(response_payload.get("response"), str):
        return _invalid_check(
            name="ollama_generate_endpoint",
            path=endpoint_url,
            message="Generate endpoint response is missing `response` text.",
            hint=hint,
        )

    return _valid_check(
        name="ollama_generate_endpoint",
        path=endpoint_url,
        message=f"Generate endpoint healthy (HTTP {status}) for model '{normalized}'.",
    )


def _ollama_checks_if_enabled(settings: Settings) -> list[DoctorCheck]:
    embedding_provider = settings.embedding_provider.lower()
    embedding_model = settings.embedding_model or _DEFAULT_EMBEDDING_MODEL
    reranker_enabled = bool(settings.reranker_enabled)
    reranker_model = settings.reranker_model or _DEFAULT_RERANKER_MODEL
    base_url = settings.ollama_base_url or _DEFAULT_OLLAMA_BASE_URL
    use_ollama = embedding_provider == "ollama" or reranker_enabled
    if not use_ollama:
        return []

    checks: list[DoctorCheck] = []
    reachability_check, models = _ollama_reachability_check(base_url)
    checks.append(reachability_check)
    ollama_reachable = reachability_check.status == "valid"

    if embedding_provider == "ollama":
        checks.append(
            _ollama_model_check(
                model_role="embedding",
                model_name=embedding_model,
                available_models=models,
            )
        )
        checks.append(
            _ollama_embeddings_health_check(
                base_url=base_url,
                model_name=embedding_model,
                ollama_reachable=ollama_reachable,
            )
        )
    if reranker_enabled:
        checks.append(
            _ollama_model_check(
                model_role="reranker",
                model_name=reranker_model,
                available_models=models,
            )
        )
        checks.append(
            _ollama_generate_health_check(
                base_url=base_url,
                model_name=reranker_model,
                ollama_reachable=ollama_reachable,
            )
        )
    return checks


def run_demo_doctor(
    *,
    run_id: str,
    input_dir: Path,
    artifact_root: Path,
    reports_dir: Path,
    matrix_reports_dir: Path,
    settings: Settings,
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
    checks.extend(_ollama_checks_if_enabled(settings))

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
