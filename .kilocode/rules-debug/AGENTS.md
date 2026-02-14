# Debug Mode Rules (Non-Obvious Only)

- `make verify` runs lint → typecheck → test in sequence; first failure stops the chain
- Single test: `uv run pytest tests/path/to/test.py::test_name -q` — `pythonpath = src` is set in `pytest.ini`
- Markers `@pytest.mark.slow` and `@pytest.mark.e2e` exist — skip with `-m "not slow"` for fast iteration
- `test_pmbok_ingestion.py` in repo root is a standalone scratch file, NOT discovered by pytest (not in `tests/`)
- Ollama-dependent tests use `pytest.importorskip("autokg_rag.eval.matrix_runner")` — they skip gracefully without the `local_llm` extra
- Config env vars all prefixed `AUTORAG_` — check `.env.example` for the full list when debugging config issues
- Artifact directories are `data/artifacts/<run_id>/` — `doctor` subcommand validates artifact structure
