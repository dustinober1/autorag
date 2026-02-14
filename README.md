# AutoRAG

## Milestone 1 quickstart

```bash
uv sync
uv run autorag smoke --input data/fixtures/pdfs --question "What is project scope?" --run-id m1
uv run pytest tests/e2e/test_m1_smoke_pipeline.py -q
```
