# Code Mode Rules (Non-Obvious Only)

- All Pydantic models MUST use `ConfigDict(extra="forbid")` — this is a project-wide convention
- Every module starts with `from __future__ import annotations` (mypy strict requires it)
- Raise `IngestError`, `SchemaError`, or `RetrievalError` from `autokg_rag.exceptions` — never bare `Exception`
- Embedding provider `"local_hash"` uses deterministic hashing (not ML) — tests rely on this for reproducibility
- The Ollama client in `src/autokg_rag/ollama/client.py` uses raw `urllib` (no `requests` dependency) — keep it that way
- Test files mirror source structure: `src/autokg_rag/kg/` → `tests/kg/`
- Tests must use `tmp_path` for artifacts and `monkeypatch` for network calls — no real Ollama in unit tests
