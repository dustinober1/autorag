# Phase 10 - CLI

## Scope
- `src/autokg_rag/cli.py`

## Findings

### 🟠 Major
1. **`eval run-matrix` prints hard-coded report paths that can be wrong**
   - File: `src/autokg_rag/cli.py:657-661`
   - Evidence: CLI response always prints `reports/experiments/...` even though `run_matrix` can write to configured overrides.
   - Impact: users can be directed to incorrect paths and assume run artifacts are missing.
   - Recommendation: return resolved paths from `run_matrix` (or compute from effective config) and print those values.

## Checklist Status
- [x] Commands surface JSON output and catch typed errors with non-zero exits
- [x] Input options use Typer validation constraints
- [ ] Help output accuracy degrades when reported artifact paths diverge from actual output location

## Strengths
- CLI wraps major flows (ingest, query, eval, judge, report) with consistent exception handling.
