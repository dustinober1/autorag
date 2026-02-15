# Phase 11 - App API & Services

## Scope
- `src/autokg_rag/app_api/service.py`
- `src/autokg_rag/app_api/endpoints.py`
- `src/autokg_rag/app_api/document_service.py`
- `src/autokg_rag/app_api/store_service.py`
- `src/autokg_rag/app_api/upload_service.py`
- `src/autokg_rag/app_api/ollama_model_service.py`
- `src/autokg_rag/app_api/doctor.py`
- `src/autokg_rag/app_api/retention.py`

## Findings

### 🟠 Major
1. **`run_id` is not constrained in app query path handling**
   - Files: `src/autokg_rag/schemas/api.py:21-24`, `src/autokg_rag/app_api/service.py:254-259`
   - Evidence: `QueryRequest.run_id` only enforces `min_length=1`; `query_service` builds `artifact_dir = settings.artifact_root / request.run_id` without canonicalization/allowlist validation.
   - Impact: crafted values like `../...` can resolve outside artifact root and read unintended directories/files reachable by the process.
   - Recommendation: apply strict run-id validation (same contract as CLI `_validated_run_id`), then verify resolved paths remain under `settings.artifact_root`.

2. **Milestone demo report uses first matrix row, not best row**
   - File: `src/autokg_rag/app_api/service.py:366-372`
   - Evidence: `_write_demo_report` sets `best_row = matrix_rows[0]`.
   - Impact: reported key metrics can misrepresent actual best-performing experiment and mislead demo decisions.
   - Recommendation: select max by primary metric (`ndcg_at_10`) before rendering report payload.

### 🟡 Minor
3. **Typecheck warnings in service-adjacent modules indicate drifting contracts**
   - Evidence command: `uv run mypy src`
   - Affected files include `src/autokg_rag/app_api/ollama_model_service.py` and `src/autokg_rag/app_api/document_service.py`.
   - Impact: strict typing guarantees are partially degraded in API/service surface.
   - Recommendation: clean all mypy diagnostics and gate merges on `make verify` green.

## Checklist Status
- [x] Service layer remains separate from Streamlit component code
- [x] File and store operations are encapsulated in dedicated services
- [ ] Path safety is incomplete for app API query `run_id` inputs
- [ ] Demo reporting currently risks inaccurate metric communication

## Strengths
- `doctor.py` and retention utilities centralize operational checks/policies.
- Service methods return typed payloads rather than ad hoc dicts.
