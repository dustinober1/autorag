# Autopilot Implementation Plan: Full Repository Code Review

## Inputs
- `plans/2026-02-14_full-repo-code-review.md`
- Repository source tree under `src/`, `app/`, `tests/`, `scripts/`, `.github/`

## Phased Execution Plan

### Phase 0 - Expansion
- Derive explicit deliverables and acceptance criteria from the source review plan.
- Write `.omx/plans/autopilot-spec.md`.

### Phase 1 - Planning
- Translate review scope into execution tasks:
  - Parallel phase audits
  - QA command execution
  - Findings synthesis
  - Consolidated reporting
- Write `.omx/plans/autopilot-impl.md`.

### Phase 2 - Execution
1. Launch parallel audit agents:
   - Agent A: phases 1-4
   - Agent B: phases 5-8
   - Agent C: phases 9-12
   - Agent D: phases 13-17
2. Collect severity-rated findings with file:line evidence.
3. Create per-phase findings files for phases 1-17.
4. Produce consolidated artifacts:
   - `issues.md`
   - `summary.md`
   - `action-items.md`

### Phase 3 - QA
- Run and record:
  - `uv run ruff check src tests`
  - `uv run mypy src`
  - `uv run pytest -q`
- Include outputs as evidence in reports.

### Phase 4 - Validation
- Run parallel validators:
  - Architecture validator (functional completeness)
  - Security validator
  - Code quality validator
- Address validation feedback by updating findings artifacts.
- Require all three to approve.

### Phase 5 - Cleanup
- Mark autopilot complete.
- Clear mode state for: `autopilot`, `ralph`, `ultrawork`, `ultraqa`.

## Definition of Done
- All required review artifacts exist under `plans/code-review-findings/`.
- Findings are phase-aligned and evidence-based.
- QA command results are current and captured.
- Validation passes from all perspectives.
- State cleaned.
