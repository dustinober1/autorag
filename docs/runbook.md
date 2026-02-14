# AutoRAG Runbook (Milestone 6)

## Reproducible demo build

```bash
uv sync
make demo-build
```

`make demo-build` runs:
1. ingest (`heading_recursive`)
2. vector indexing (`bge-small-en-v1.5`)
3. KG build
4. hybrid query + grounded answer payload sample
5. lightweight eval dataset + matrix report
6. milestone report generation

Default outputs:
- `data/artifacts/m6/demo_payload_samples.jsonl`
- `reports/milestones/m6_demo_report.md`
- `reports/experiments/matrix_results.csv`
- `reports/experiments/matrix_results.json`
- `reports/experiments/leaderboard.md`

## M7 hardening commands

```bash
make verify
make demo-smoke
uv run autorag doctor
```

- `make verify` runs lint + typecheck + tests in CI-equivalent order.
- `make demo-smoke` runs a fast-path M6 smoke build (defaults `run_id=m6_smoke`, `mode=vector`, `top_k=4`).
- `autorag doctor` validates local demo prerequisites and output artifacts, returning actionable missing-file hints.

## Interactive app

```bash
uv run streamlit run app/streamlit_app.py
```

In the app:
1. choose `run_id`
2. select mode (`vector`, `graph`, or `hybrid`)
3. run a question
4. inspect answer + citation viewer

## Configurable demo-build environment variables

- `AUTORAG_ARTIFACT_ROOT` (default: `data/artifacts`)
- `AUTORAG_DEMO_RUN_ID` (default: `m6`)
- `AUTORAG_DEMO_INPUT_DIR` (default: `data/fixtures/pdfs`)
- `AUTORAG_DEMO_QUESTION` (default: `Compare mitigation and acceptance strategies.`)
- `AUTORAG_DEMO_MODE` (default: `hybrid`)
- `AUTORAG_DEMO_TOP_K` (default: `8`)
- `AUTORAG_DEMO_REPORTS_DIR` (default: `reports/milestones`)
- `AUTORAG_DEMO_MATRIX_REPORTS_DIR` (default: `reports/experiments`)
