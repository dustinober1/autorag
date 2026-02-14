from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from autokg_rag.config.settings import Settings

CHUNKING_STRATEGIES = (
    "fixed_400_50",
    "heading_recursive",
    "sentence_window_5",
    "semantic_breakpoint",
)
EMBEDDING_MODELS = (
    "bge_small",
    "minilm_l6",
    "e5_small",
)
RETRIEVAL_MODES = (
    "vector",
    "graph",
    "hybrid",
)


def _resolve_callable(module: Any) -> Any:
    candidate_names = (
        "build_factorial_grid",
        "build_experiment_grid",
        "plan_experiments",
        "generate_experiment_grid",
        "emit_factorial_grid",
        "run_matrix",
        "run_experiment_matrix",
    )
    for name in candidate_names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn

    raise AssertionError(
        "Could not find matrix grid callable in autokg_rag.eval.matrix_runner."
    )


def _call_with_supported_kwargs(fn: Any, provided: dict[str, Any]) -> Any:
    signature = inspect.signature(fn)
    aliases = {
        "config_file": "config_path",
        "matrix_config_path": "config_path",
        "question_path": "questions_path",
        "questions_file": "questions_path",
        "questions_jsonl": "questions_path",
        "report_path": "output_dir",
        "reports_dir": "output_dir",
    }

    kwargs: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        provided_key = aliases.get(parameter.name, parameter.name)
        if provided_key in provided:
            kwargs[parameter.name] = provided[provided_key]
            continue

        if parameter.default is inspect.Parameter.empty:
            raise AssertionError(
                f"Unsupported required matrix parameter '{parameter.name}' for "
                f"{fn.__module__}.{fn.__name__}."
            )

    return fn(**kwargs)


def _extract_exp_id(row: Any) -> str | None:
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        if "exp_id" in row:
            return str(row["exp_id"])
        if {"chunking", "embedding", "retrieval"}.issubset(row):
            return f"exp_{row['chunking']}_{row['embedding']}_{row['retrieval']}"
        return None
    if hasattr(row, "exp_id"):
        return str(row.exp_id)
    if hasattr(row, "experiment_id"):
        return str(row.experiment_id)
    if hasattr(row, "model_dump"):
        dumped = row.model_dump(mode="json")
        return _extract_exp_id(dumped)
    return None


def _exp_ids_from_return_value(value: Any) -> set[str]:
    collected: set[str] = set()
    if isinstance(value, list):
        for item in value:
            exp_id = _extract_exp_id(item)
            if exp_id:
                collected.add(exp_id)
    elif isinstance(value, dict):
        for key in ("rows", "experiments", "results"):
            if key in value and isinstance(value[key], list):
                for item in value[key]:
                    exp_id = _extract_exp_id(item)
                    if exp_id:
                        collected.add(exp_id)
        exp_id = _extract_exp_id(value)
        if exp_id:
            collected.add(exp_id)
    else:
        exp_id = _extract_exp_id(value)
        if exp_id:
            collected.add(exp_id)
    return collected


def _exp_ids_from_written_artifacts(root: Path) -> set[str]:
    exp_ids: set[str] = set()

    json_candidates = sorted(root.rglob("matrix_results.json"))
    for path in json_candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("rows", [])
            if isinstance(rows, list):
                for row in rows:
                    exp_id = _extract_exp_id(row)
                    if exp_id:
                        exp_ids.add(exp_id)

    csv_candidates = sorted(root.rglob("matrix_results.csv"))
    for path in csv_candidates:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                exp_id = _extract_exp_id(row)
                if exp_id:
                    exp_ids.add(exp_id)

    per_query_files = sorted(root.rglob("*.jsonl"))
    for path in per_query_files:
        if "per_query" in path.parts:
            exp_ids.add(path.stem)

    return exp_ids


def test_matrix_runner_emits_complete_factorial_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix_runner = pytest.importorskip("autokg_rag.eval.matrix_runner")
    runner_fn = _resolve_callable(matrix_runner)

    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "configs" / "experiments" / "matrix.yaml"
    questions_path = tmp_path / "eval" / "datasets" / "questions.jsonl"
    output_dir = tmp_path / "reports" / "experiments"

    config_path.parent.mkdir(parents=True, exist_ok=True)
    questions_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions_path.write_text(
        json.dumps(
            {
                "question_id": "q001",
                "type": "fact",
                "question": "Deterministic matrix fixture question?",
                "gold_citations": [
                    {
                        "doc_id": "fixture_doc",
                        "page": 1,
                        "section": "Fixture",
                        "chunk_id": "fixture_chunk_001",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    matrix_config = {
        "factors": {
            "chunking": list(CHUNKING_STRATEGIES),
            "embedding": list(EMBEDDING_MODELS),
            "retrieval": list(RETRIEVAL_MODES),
        },
        "chunking": list(CHUNKING_STRATEGIES),
        "embeddings": list(EMBEDDING_MODELS),
        "retrieval_modes": list(RETRIEVAL_MODES),
        "dataset_path": str(questions_path),
        "questions_path": str(questions_path),
        "metrics": {"primary": "nDCG@10"},
    }
    config_path.write_text(yaml.safe_dump(matrix_config, sort_keys=True), encoding="utf-8")

    result = _call_with_supported_kwargs(
        runner_fn,
        {
            "artifact_root": tmp_path / "artifacts",
            "config": matrix_config,
            "config_path": config_path,
            "dry_run": True,
            "factors": matrix_config["factors"],
            "output_dir": output_dir,
            "questions_path": questions_path,
            "run_id": "m5_grid",
            "settings": Settings(artifact_root=tmp_path / "artifacts"),
        },
    )

    observed_exp_ids = _exp_ids_from_return_value(result)
    if not observed_exp_ids:
        observed_exp_ids = _exp_ids_from_written_artifacts(tmp_path)

    expected_exp_ids = {
        f"exp_{chunking}_{embedding}_{retrieval}"
        for chunking in CHUNKING_STRATEGIES
        for embedding in EMBEDDING_MODELS
        for retrieval in RETRIEVAL_MODES
    }
    assert len(expected_exp_ids) == 36
    assert expected_exp_ids.issubset(observed_exp_ids)


def test_matrix_runner_supports_optional_ollama_factors() -> None:
    matrix_runner = pytest.importorskip("autokg_rag.eval.matrix_runner")
    build_factorial_grid = matrix_runner.build_factorial_grid

    matrix_config = {
        "factors": {
            "chunking": ["heading_recursive"],
            "embedding_model": ["embeddinggemma:300m"],
            "retrieval": ["hybrid"],
            "embedding_provider": ["local_hash", "ollama"],
            "reranker_enabled": ["false", "true"],
            "reranker_model": ["llama3:8b"],
        },
    }
    grid = build_factorial_grid(matrix_config)

    assert len(grid) == 4
    assert {spec.embedding_provider for spec in grid} == {"local_hash", "ollama"}
    assert {spec.reranker_enabled for spec in grid} == {False, True}
    assert {spec.embedding_model for spec in grid} == {"embeddinggemma:300m"}
    assert {spec.reranker_model for spec in grid} == {"llama3:8b"}

    exp_ids = {spec.exp_id for spec in grid}
    assert "exp_heading_recursive_embeddinggemma_300m_hybrid" in exp_ids
    assert any("_provider_ollama" in exp_id for exp_id in exp_ids)
    assert any("_reranker_llama3_8b" in exp_id for exp_id in exp_ids)
