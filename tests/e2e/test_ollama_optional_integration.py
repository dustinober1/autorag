from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

import pytest


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _model_variants(model_name: str) -> set[str]:
    normalized = model_name.strip()
    if not normalized:
        return set()
    variants = {normalized}
    if ":" in normalized:
        variants.add(normalized.split(":", 1)[0])
    else:
        variants.add(f"{normalized}:latest")
    return variants


def _ollama_models(base_url: str) -> set[str] | None:
    url = f"{base_url.rstrip('/')}/api/tags"
    req = urllib_request.Request(url=url, method="GET")
    try:
        with urllib_request.urlopen(req, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, OSError, ValueError):
        return None

    if not isinstance(payload, dict):
        return set()

    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return set()

    models: set[str] = set()
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            value = item.get(key)
            if not isinstance(value, str):
                continue
            models.update(_model_variants(value))
    return models


def test_doctor_ollama_optional_integration(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    base_url = os.getenv("AUTORAG_OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model = os.getenv("AUTORAG_EMBEDDING_MODEL", "embeddinggemma:300m")

    models = _ollama_models(base_url)
    if models is None:
        pytest.skip("Ollama API unavailable; skipping optional integration test.")
    if not (_model_variants(embedding_model) & models):
        pytest.skip(
            f"Ollama model '{embedding_model}' not available; run `ollama pull {embedding_model}`."
        )

    run_id = "m6_ollama_optional"
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_file(input_dir / "fixture.pdf", "fixture text")

    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / run_id
    _write_file(run_dir / "chunks.parquet", "ok")
    _write_file(run_dir / "embeddings.npy", "ok")
    _write_file(run_dir / "embedding_meta.parquet", "ok")
    _write_file(run_dir / "kg.sqlite", "ok")
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
        "run_id,exp_id\n"
        "m6_ollama_optional,exp_1\n",
    )
    _write_file(
        matrix_reports_dir / "matrix_results.json",
        (
            '{"run_id":"m6_ollama_optional","summary":{"total_experiments":1},'
            '"rows":[{"run_id":"m6_ollama_optional","exp_id":"exp_1"}]}'
        ),
    )
    _write_file(matrix_reports_dir / "leaderboard.md", "# Leaderboard")

    env = os.environ.copy()
    env["AUTORAG_ARTIFACT_ROOT"] = str(artifact_root)
    env["AUTORAG_DEMO_RUN_ID"] = run_id
    env["AUTORAG_DEMO_INPUT_DIR"] = str(input_dir)
    env["AUTORAG_DEMO_REPORTS_DIR"] = str(reports_dir)
    env["AUTORAG_DEMO_MATRIX_REPORTS_DIR"] = str(matrix_reports_dir)
    env["AUTORAG_EMBEDDING_PROVIDER"] = "ollama"
    env["AUTORAG_EMBEDDING_MODEL"] = embedding_model
    env["AUTORAG_RERANKER_ENABLED"] = "false"
    env["AUTORAG_OLLAMA_BASE_URL"] = base_url

    result = subprocess.run(
        ["uv", "run", "autorag", "doctor"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    report = json.loads(result.stdout)
    checks = report.get("checks")
    assert isinstance(checks, list)
    checks_by_name = {
        str(row.get("name")): row
        for row in checks
        if isinstance(row, dict)
    }
    for check_name in (
        "ollama_reachability",
        "ollama_model_embedding",
        "ollama_embeddings_endpoint",
    ):
        row = checks_by_name.get(check_name)
        assert isinstance(row, dict), check_name
        assert row.get("status") == "valid", check_name
