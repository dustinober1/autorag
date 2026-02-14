# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Commands
- **Lint/typecheck/test**: `make verify` (runs `uv run ruff check src tests`, `uv run mypy src`, `uv run pytest -q`)
- **Single test**: `uv run pytest tests/path/to/test_file.py::test_function_name -q`
- **Install deps**: `uv sync --extra dev` (add `--extra vision` or `--extra local_llm` as needed)
- **CLI entry**: `uv run autorag <subcommand>` (maps to `src/autokg_rag/cli.py:main`)

## Code Style (non-obvious)
- Line length 100 (`ruff.toml`), ruff selects `E,F,I,B,UP,N,W` — isort is enforced via `I`
- mypy strict mode enabled — all functions need return type annotations and `from __future__ import annotations`
- All Pydantic models use `ConfigDict(extra="forbid")` — adding unexpected fields raises errors
- Source lives in `src/autokg_rag/` but package imports are `from autokg_rag.xxx import ...` (src-layout via `tool.setuptools.package-dir`)
- Custom exceptions in `src/autokg_rag/exceptions.py`: `AutoRAGError` → `IngestError`, `SchemaError`, `RetrievalError` — use these, not bare exceptions

## Architecture Gotchas
- Artifacts written to `data/artifacts/<run_id>/` — each milestone (m1–m6) gets its own run-id
- Config layering: YAML files in `configs/` → env vars prefixed `AUTORAG_` → CLI flags (see `src/autokg_rag/config/loaders.py`)
- `embedding_provider="local_hash"` is a deterministic test-friendly stub, NOT a real embedding — switch to `"ollama"` for actual vectors
- Tests use `tmp_path` extensively and `monkeypatch` to avoid real Ollama/network calls
- `test_pmbok_ingestion.py` in repo root is a standalone scratch test, not part of the test suite

## Commit Rules
- Commit at the file level for each change — do not batch commits. Include issue number in message when applicable.
