# Phase 16 - Scripts & CI

## Scope
- `scripts/*.sh`
- `.github/workflows/*`

## Findings

### 🔵 Suggestion
1. **CI does not validate shell scripts in `scripts/`**
   - File: `.github/workflows/ci.yml:27-31`
   - Evidence: workflow installs deps and runs only `make verify` (lint/type/test).
   - Impact: script syntax/behavior regressions can ship undetected until manual runs.
   - Recommendation: add `bash -n scripts/*.sh` (or a lightweight smoke target) to CI.

## Checklist Status
- [x] CI runs lint, typecheck, and tests
- [x] Scripts reviewed use `set -euo pipefail` patterns in key paths
- [ ] CI script validation coverage is currently missing

## Strengths
- CI pipeline is simple and fast with `uv` caching.
- Script quality baseline is good in sampled milestone scripts.
