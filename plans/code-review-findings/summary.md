# Full Repository Code Review - Summary

## Scope Completed
Executed all 17 phases defined in `plans/2026-02-14_full-repo-code-review.md` and generated per-phase findings under `plans/code-review-findings/phase-*.md`.

## Overall Assessment
- **Runtime correctness baseline:** reasonable (test suite passes).
- **Static quality baseline:** currently not acceptable for strict-gated workflow (ruff + mypy failing).
- **Highest-risk functional areas:** ingestion metadata fidelity, graph retrieval directionality, eval artifact correctness, arXiv download hardening.

## Severity Tally
- Critical: 0
- Major: 10
- Minor: 5
- Suggestions: 2

## Top Risks To Address First
1. Quality-gate breakage (`ruff` + `mypy`) undermines reliability of merges.
2. App API query flow can traverse outside artifact root via unsanitized `run_id`.
3. Graph retrieval can fail for valid target-focused questions.
4. Eval matrix per-query outputs can be overwritten due to `exp_id` collisions.
5. Demo metrics can misreport performance by selecting first row rather than best.
6. arXiv integration writes unvalidated payloads and may reuse stale local files.

## Notable Strengths
- Architecture is modular across library, app API, eval pipeline, and Streamlit UI.
- Test breadth is strong across multiple subsystem boundaries.
- Structured logging/metrics are consistently present.
- CLI exception handling and typed contracts are broadly consistent.

## QA Snapshot (Current Run)
- `ruff`: fail
- `mypy`: fail
- `pytest`: pass (`82 passed`, `4 warnings`)
