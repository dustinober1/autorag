# Debug Mode Rules (Non-Obvious Only)

- `@pytest.mark.slow` and `@pytest.mark.e2e` markers exist in `pytest.ini` — skip with `-m "not slow"`
- `test_pmbok_ingestion.py` in repo root is NOT part of test suite (standalone scratch file)
- Use `pytest.importorskip` pattern for Ollama tests
- Artifacts written to `data/artifacts/<run_id>/` — each milestone gets unique run-id
- `uv run autorag doctor` validates artifacts
