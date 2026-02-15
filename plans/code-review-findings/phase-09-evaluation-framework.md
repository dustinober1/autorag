# Phase 09 - Evaluation Framework

## Scope
- `src/autokg_rag/eval/dataset_builder.py`
- `src/autokg_rag/eval/matrix_runner.py`
- `src/autokg_rag/eval/metrics.py`
- `src/autokg_rag/eval/judge.py`
- `src/autokg_rag/eval/ab_test.py`
- `src/autokg_rag/eval/report.py`

## Findings

### 🟠 Major
1. **Experiment ID collisions when `reranker_enabled=false` can overwrite per-query artifacts**
   - File: `src/autokg_rag/eval/matrix_runner.py:278-293`, `src/autokg_rag/eval/matrix_runner.py:679-705`, `src/autokg_rag/eval/matrix_runner.py:620-627`
   - Evidence: factorial grid iterates over all `reranker_model` values, but `_format_experiment_id` excludes reranker suffix when disabled.
   - Impact: multiple logical runs can share identical `exp_id`; per-query JSONL files are overwritten by later runs.
   - Recommendation: either skip reranker-model factor when disabled or include reranker model token in `exp_id` regardless of enablement.

## Checklist Status
- [x] Core matrix/eval pipeline has broad test coverage
- [x] Metric outputs are generated in CSV/JSON/per-query forms
- [ ] Per-query artifact uniqueness under optional factors is not protected by tests

## Strengths
- End-to-end eval harness (`tests/e2e/test_m5_eval_harness.py`) validates matrix/report artifact generation.
- Metrics stack includes recall, nDCG, citation precision, and faithfulness proxy.
