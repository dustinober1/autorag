# Phase 15 - Test Suite Review

## Scope
- `tests/*`
- `test_pmbok_ingestion.py`

## Findings

### 🟡 Minor
1. **Root test file uses return values instead of assertions (pytest warnings)**
   - File: `test_pmbok_ingestion.py:19-139`
   - Evidence command: `uv run pytest -q` emitted 4 `PytestReturnNotNoneWarning` warnings for returning `bool` from test functions.
   - Impact: tests can appear to pass while not asserting behavior idiomatically; warning noise can hide real issues.
   - Recommendation: convert to assertion-based tests in `tests/` and/or mark as utility script outside pytest collection.

2. **Coverage gaps for discovered failure modes**
   - Missing/weak tests:
     - No regression test for graph retrieval when query matches only target-side nodes (`src/autokg_rag/kg/retriever.py`).
     - No test guarding matrix per-query artifact collisions when `reranker_enabled=false` and multiple reranker models exist.
     - No unit test for `_write_demo_report` selecting best metric row.
   - Impact: identified defects can regress unnoticed.
   - Recommendation: add focused unit tests for each known failure mode.

## Checklist Status
- [x] Suite is broad and currently passing (`82 passed`)
- [x] E2E harnesses exist for milestone flows
- [ ] Warnings indicate non-idiomatic test patterns in root test module
- [ ] Some critical edge cases are untested

## Strengths
- Test coverage spans ingestion, retrieval, eval, API, and e2e scenarios.
- CI executes the test suite via `make verify`.
