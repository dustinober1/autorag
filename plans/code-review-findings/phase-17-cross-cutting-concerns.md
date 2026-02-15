# Phase 17 - Cross-Cutting Concerns

## Scope
- Error handling, logging, security, performance, docs, dead code, dependency hygiene, duplication

## Findings

### 🟠 Major
1. **Quality-gate mismatch: test suite passes but static gates fail**
   - Evidence commands:
     - `uv run ruff check src tests` -> 37 violations
     - `uv run mypy src` -> 9 errors
     - `uv run pytest -q` -> 82 passed, 4 warnings
   - Impact: repository can appear healthy through runtime tests while CI-quality gates are red.
   - Recommendation: prioritize lint/type debt burn-down and enforce green `make verify` before merge.

2. **App-facing query path assembly is vulnerable to run-id path traversal**
   - Files: `src/autokg_rag/schemas/api.py:21-24`, `src/autokg_rag/app_api/service.py:254-259`
   - Evidence: user-provided `run_id` is only `min_length=1`, then joined directly to `settings.artifact_root`.
   - Impact: crafted `run_id` can resolve outside artifact root and expose unintended filesystem data paths.
   - Recommendation: enforce run-id allowlist/regex and verify resolved path ancestry against artifact root.

### 🟡 Minor
3. **Root-level `test_pmbok_ingestion.py` behaves like a mixed script/test artifact**
   - File: `test_pmbok_ingestion.py:1-163`
   - Impact: unclear ownership and execution semantics (manual script + pytest-collected tests) create maintenance ambiguity.
   - Recommendation: move to `tests/` as proper tests or relocate to `scripts/` as explicit diagnostic utility.

## Checklist Status
- [x] Structured logging and metrics are present across pipelines
- [x] Security-sensitive path fields are generally sanitized at boundaries (e.g., arXiv filename normalization)
- [ ] App query `run_id` path boundary checks are missing
- [ ] Prompt/file/network hardening remains incomplete in specific flows (see Phases 08 and 13)
- [ ] Dependency policy remains lower-bound, not reproducibly pinned at metadata level

## Strengths
- Architecture separation (core library, app API, UI, eval) is clear.
- Error types (`AutoRAGError` subclasses) are used consistently across major modules.
