# Code Mode Rules (Non-Obvious Only)

- All Pydantic models MUST use `ConfigDict(extra="forbid")` — adding unexpected fields raises errors
- `embedding_provider="local_hash"` is a deterministic test-friendly stub (NOT ML embeddings) — use `"ollama"` for real vectors
- Ollama client in `src/autokg_rag/ollama/client.py` uses raw `urllib` (no `requests` dependency)
- Tests MUST use `tmp_path` + `monkeypatch` to avoid real Ollama/network calls
- Custom exceptions in `src/autokg_rag/exceptions.py`: `AutoRAGError` → `IngestError`, `SchemaError`, `RetrievalError` — use these, NOT bare exceptions
