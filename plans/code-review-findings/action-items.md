# Recommended Action Items

## P0 - Immediate (Blocker-Level)
1. **Restore green quality gates**
   - Resolve all current mypy and ruff errors.
   - Add/keep CI policy requiring `make verify` pass before merge.
2. **Block `run_id` path traversal in app query flow**
   - Validate `run_id` against strict allowlist/regex at API boundary.
   - Confirm resolved `artifact_dir` is inside `settings.artifact_root`.
   - Add regression tests for traversal attempts (`../`, absolute paths, encoded variants).
3. **Fix graph retriever directionality**
   - Implement bidirectional or reverse-edge traversal support in `src/autokg_rag/kg/retriever.py`.
   - Add tests for target-node-only query terms.
4. **Fix matrix artifact collision risk**
   - Ensure unique experiment IDs even when reranker is disabled.
   - Add regression tests validating per-query file uniqueness.

## P1 - High Value
5. **Correct demo metric selection**
   - Select best row by primary metric in `_write_demo_report`.
   - Add unit test for deterministic best-row selection.
6. **Harden arXiv download pipeline**
   - Validate content type/size/signature before write.
   - Add redownload/freshness policy and stale-file cleanup strategy.
7. **Harden config parse failure UX**
   - Wrap YAML parse exceptions with `AutoRAGError` including file context.

## P2 - Quality Improvements
8. **Constrain schema enums/taxonomy**
   - Convert `ChunkRecord.chunk_type` to enum/Literal.
9. **Preserve cross-reference kinds**
   - Emit Figure/Table/Chapter labels instead of collapsing to Section.
10. **Improve answer sentence selection**
   - Rank/select best supporting sentence per chunk instead of always first sentence.
11. **Normalize root PMBOK test artifact**
   - Convert to proper pytest tests or relocate as script utility.

## P3 - Tooling/Resilience
12. **Add retry/backoff controls to Ollama client**
13. **Add CI validation for shell scripts**
    - e.g., `bash -n scripts/*.sh` and optionally one lightweight smoke execution.

## Suggested Implementation Order
1. P0 items (quality gates, graph retrieval, matrix artifact integrity)
2. P1 items (demo correctness, arXiv hardening, config UX)
3. P2/P3 cleanup with added regression tests
