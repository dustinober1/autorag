# Autopilot Spec: Full Repository Code Review

## Context
- **Date:** 2026-02-14
- **Repository:** `/Users/dustinober/Projects/AutoRAG`
- **Source Plan:** `plans/2026-02-14_full-repo-code-review.md`
- **Execution Mode Requested:** `autopilot` with additional necessary skills

## Problem Statement
Produce a complete, evidence-based code review across the entire AutoRAG repository, with findings organized by the 17 review phases defined in the source plan.

## Goals
1. Execute every review phase from Foundation through Cross-Cutting concerns.
2. Produce line-referenced findings with clear severity and actionable remediation.
3. Generate all deliverables specified by the plan:
   - `plans/code-review-findings/phase-NN-*.md`
   - `plans/code-review-findings/issues.md`
   - `plans/code-review-findings/summary.md`
   - `plans/code-review-findings/action-items.md`
4. Include fresh QA signals (`ruff`, `mypy`, `pytest`) in the review evidence.

## Non-Goals
- Refactoring source code in this run.
- Fixing discovered issues in this run.
- Redesigning architecture beyond review recommendations.

## Review Requirements
- Use severity levels: Critical, Major, Minor, Suggestion.
- Reference exact files and lines for each finding.
- Cover security, correctness, maintainability, performance, and test quality.
- Include checklist status per phase (pass/gap notes).

## Constraints
- Repository may already contain unrelated uncommitted changes; do not revert user work.
- Findings must be substantiated by code evidence or executed tool output.
- Keep outputs in markdown under `plans/code-review-findings/`.

## Acceptance Criteria
- All 17 phases have findings documents.
- Consolidated issues list exists with severity grouping.
- Summary and prioritized action plan are present.
- QA evidence from current run is included.
- Autopilot lifecycle state is transitioned and cleaned on completion.
