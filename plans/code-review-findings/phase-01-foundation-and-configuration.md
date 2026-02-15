# Phase 01 - Foundation & Configuration

## Scope
- `pyproject.toml`, `Makefile`, `ruff.toml`, `mypy.ini`, `pytest.ini`
- `src/autokg_rag/config/settings.py`
- `src/autokg_rag/config/loaders.py`
- `src/autokg_rag/io/artifacts.py`
- `src/autokg_rag/observability/logging.py`, `src/autokg_rag/observability/metrics.py`
- `configs/*.yaml`

## Findings

### 🟠 Major
1. **Malformed YAML is not wrapped in a user-facing config error**
   - File: `src/autokg_rag/config/loaders.py:80`
   - Evidence: `yaml.safe_load(...)` is called directly without `try/except`; parse failures raise raw YAML exceptions.
   - Impact: CLI/config boot can fail with an unstructured traceback instead of consistent `AutoRAGError` handling.
   - Recommendation: catch `yaml.YAMLError` and raise `AutoRAGError` with path/context.

2. **Strict typecheck gate currently fails**
   - Evidence command: `uv run mypy src`
   - Current result: 9 errors in `src/autokg_rag/arxiv/client.py`, `src/autokg_rag/app_api/ollama_model_service.py`, `src/autokg_rag/app_api/document_service.py`.
   - Impact: declared quality gate (`strict = True`) is not currently satisfied.
   - Recommendation: fix unresolved typed import usage for `arxiv`, remove stale `type: ignore`, and resolve overload/no-any-return diagnostics.

### 🟡 Minor
3. **Dependency declarations are lower-bound ranges, not reproducible pins**
   - File: `pyproject.toml:13-44`
   - Evidence: dependencies use `>=` for runtime and dev packages.
   - Impact: upstream releases can change behavior between installs; lockfile helps locally, but package metadata remains open-ended.
   - Recommendation: keep lockfile authoritative for CI and document install workflow; pin or constrain upper bounds for critical runtime packages.

4. **Lint gate currently fails in `src/autokg_rag/ingest/*` and schema formatting**
   - Evidence command: `uv run ruff check src tests`
   - Current result: 37 violations (import sorting, long lines, deprecated typing aliases, whitespace).
   - Impact: `make verify` fails in environments enforcing lint as a hard gate.
   - Recommendation: run `ruff --fix` for mechanical issues and manually resolve remaining long-line/type-alias issues.

## Checklist Status
- [x] `mypy --strict` configured
- [ ] `mypy --strict` currently clean
- [x] YAML missing-file handling is resilient (`path.exists()` guard)
- [ ] YAML malformed-file handling is resilient
- [x] Config defaults are present and explicit in `configs/base.yaml`
- [x] Artifact I/O uses explicit paths and structured parsing errors

## Strengths
- `mypy.ini` is genuinely strict (`strict = True`, redundant/unused warnings enabled).
- `Makefile` includes unified `verify` target (`lint`, `typecheck`, `test`).
- Observability modules emit structured records (`JSONL`) and timing/counter metrics consistently.
