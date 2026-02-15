# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Package & Imports
- Package is `autorag` on PyPI but imports are `from autokg_rag.xxx import ...` (src-layout in pyproject.toml)
- CLI entry point is `autorag = "autokg_rag.cli:main"` (NOT `autokg_rag`)

## Code Style
- All Pydantic models MUST use `ConfigDict(extra="forbid")` — adding unexpected fields raises errors
- Custom exceptions in `src/autokg_rag/exceptions.py`: `AutoRAGError` → `IngestError`, `SchemaError`, `RetrievalError` — use these, not bare exceptions

## Architecture Gotchas
- Artifacts written to `data/artifacts/<run_id>/` — each milestone (m1–m6) gets its own run-id
- Config layering: YAML files in `configs/` → env vars prefixed `AUTORAG_` → CLI flags (see `src/autokg_rag/config/loaders.py`)
- `embedding_provider="local_hash"` is a deterministic test-friendly stub, NOT a real embedding — switch to `"ollama"` for actual vectors

## Testing
- Tests use `tmp_path` extensively and `monkeypatch` to avoid real Ollama/network calls
- `test_pmbok_ingestion.py` in repo root is a standalone scratch test, not part of the test suite

## Commit Rules
- Commit at the file level for each change — do not batch commits. Include issue number in message when applicable.
