# AutoRAG

## Quickstart

```bash
uv sync
```

## Milestone command table

| Milestone | Goal | Primary command | Acceptance test |
| --- | --- | --- | --- |
| M1 | Smoke pipeline with cited answer | `make m1` | `uv run pytest tests/e2e/test_m1_smoke_pipeline.py -q` |
| M2 | Ingest + chunking + vector baseline | `make m2` | `uv run pytest tests/e2e/test_m2_vector_pipeline.py -q` |
| M3 | Automated ontology + graph retrieval | `make m3` | `uv run pytest tests/e2e/test_m3_graph_pipeline.py -q` |
| M4 | Hybrid retrieval + grounded answer | `make m4` | `uv run pytest tests/e2e/test_m4_hybrid_qa.py -q` |
| M5 | Eval dataset + matrix + leaderboard | `make m5` | `uv run pytest tests/e2e/test_m5_eval_harness.py -q` |
| M6 | Portfolio demo build + app | `make demo-build` | `uv run pytest tests/e2e/test_m6_demo_workflow.py -q` |

## Demo app

```bash
make demo-build
uv run streamlit run app/streamlit_app.py
```

### New App Capabilities

- Store CRUD from the sidebar (`Create Store`, `Delete Selected Store`)
- Multi-document PDF upload into a selected store
- Per-document listing, incremental add, and remove with embedding row realignment
- Dynamic Ollama model discovery from `/api/tags` with runtime model selection
- arXiv search, select, download, and ingest directly into a target store

### Typical UI Flow

1. Create or select a store in the sidebar.
2. Upload PDFs in `Upload Documents` (or use `arXiv Search & Import`).
3. Optionally adjust answer/embedding/reranker model selections.
4. Ask a question and inspect citations, retrieval hits, and grounding trace.

## Optional Ollama Demo Path

Default local/CI flow remains `make demo-build`.  
For the optional Ollama path:

```bash
ollama serve
ollama pull embeddinggemma:300m
ollama pull llama3:8b
make demo-build-ollama
```

Matrix sample config: `configs/experiments/matrix_ollama.yaml`

Runbook: `docs/runbook.md`
