# Milestones

## M1 - Skeleton + contracts + observability

- Runner: `make m1`
- Validation test: `uv run pytest tests/e2e/test_m1_smoke_pipeline.py -q`
- Done when: smoke answer is non-empty and citations map to existing chunks.

## M2 - Ingest + chunking + vector retrieval

- Runner: `make m2`
- Validation test: `uv run pytest tests/e2e/test_m2_vector_pipeline.py -q`
- Done when: ingest/index/query complete and vector hits include provenance.

## M3 - Ontology extraction + graph retrieval

- Runner: `make m3`
- Validation test: `uv run pytest tests/e2e/test_m3_graph_pipeline.py -q`
- Done when: KG artifacts exist and graph query returns cited answer evidence.

## M4 - Hybrid retrieval + grounded answer

- Runner: `make m4` (or `bash scripts/run_m4_pipeline.sh`)
- Validation test: `uv run pytest tests/e2e/test_m4_hybrid_qa.py -q`
- Done when: `hybrid_hits.jsonl`, `answers.jsonl`, and `citation_trace.jsonl` are emitted.

## M5 - Eval harness + experiment matrix

- Runner: `make m5` (or `bash scripts/run_m5_eval.sh`)
- Validation test: `uv run pytest tests/e2e/test_m5_eval_harness.py -q`
- Done when: question dataset, matrix CSV/JSON, per-query JSONL, and leaderboard are generated.

## M6 - Demo app + runbook workflow

- Runner: `make demo-build`
- App: `uv run streamlit run app/streamlit_app.py`
- Validation test: `uv run pytest tests/e2e/test_m6_demo_workflow.py -q`
- Done when: demo payload/report are generated and app mode switching shows citations.
