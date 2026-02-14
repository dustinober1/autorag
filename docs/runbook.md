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

## Deterministic demo fixture pack (M8 kickoff)

- Fixture pack definition: `configs/demo_fixture_pack.yaml`
- Source set is fixed to:
  - `data/fixtures/pdfs/project_scope_fixture.pdf`
  - `data/fixtures/pdfs/risk_response_fixture.pdf`
- Selection policy:
  - no random sampling
  - lexical filename ordering
  - include only files listed in the manifest
  - SHA-256 checksums are pinned in the manifest

## M7 hardening commands

```bash
make verify
make demo-smoke
uv run autorag doctor
```

- `make verify` runs lint + typecheck + tests in CI-equivalent order.
- `make demo-smoke` runs a fast-path M6 smoke build (defaults `run_id=m6_smoke`, `mode=vector`, `top_k=4`).
- `autorag doctor` validates local demo prerequisites and output artifacts with `valid` / `missing` / `invalid` checks.

## Failure triage flow (M8 kickoff)

1. Run quality gates in this order:
   ```bash
   make verify
   make demo-smoke
   uv run autorag doctor --run-id m6_smoke
   ```
2. If `make verify` fails:
   - fix lint/typecheck/test failures first; do not trust downstream artifact checks yet.
3. If `make demo-smoke` fails:
   - check fixture availability in `data/fixtures/pdfs`
   - confirm `AUTORAG_DEMO_*` environment overrides point to valid paths
4. If `autorag doctor` reports `missing`:
   - re-run `make demo-build` or `make demo-smoke` for the target `run_id`
   - confirm artifact/report directories are correct for that run
5. If `autorag doctor` reports `invalid`:
   - regenerate the specific artifact and inspect content contract:
     - `answers.jsonl`: JSONL rows with `answer_text` + non-empty `citations`
     - `demo_payload_samples.jsonl`: JSONL rows with `question` + `answer_record.answer_text`
     - `matrix_results.json`: object with `run_id`, `summary`, non-empty `rows`
     - `matrix_results.csv`: header + at least one data row
     - `m6_demo_report.md` and `leaderboard.md`: non-empty markdown files

## Artifact retention helper (safe default)

`cleanup-artifacts` is non-destructive by default and returns a dry-run plan.

```bash
uv run autorag cleanup-artifacts --keep-latest 3 --keep-run m6 --keep-run m6_smoke
```

Apply deletion only when ready:

```bash
uv run autorag cleanup-artifacts --keep-latest 3 --keep-run m6 --keep-run m6_smoke --apply
```

## Known-good run snapshot template

- Template path: `reports/known_good_run_snapshot.template.md`
- Copy it when a run is fully green to record reproducibility details (inputs, commands, gates, and artifact checks).

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
