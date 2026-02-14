# Autopilot Implementation Plan: Milestone 3

## Phase 2 Execution Tasks

1. Schemas and config updates
- Extend `src/autokg_rag/schemas/records.py` with KG node/edge records.
- Export new schemas in `src/autokg_rag/schemas/__init__.py`.

2. KG module implementation
- Add `src/autokg_rag/kg/ontology_extract.py` for deterministic extraction + canonicalization mapping.
- Add `src/autokg_rag/kg/store_sqlite.py` for sqlite schema creation and persistence.
- Add `src/autokg_rag/kg/retriever.py` for multihop traversal and `RetrievalHitRecord` generation.
- Add `src/autokg_rag/kg/pipeline.py` for `build-kg` orchestration and artifact writing.
- Add `src/autokg_rag/kg/__init__.py` exports.

3. CLI integration
- Add `build-kg` command.
- Extend `query` command to support `--mode graph` using KG retriever.
- Keep vector mode behavior unchanged.

4. Tests
- Add KG extraction unit test.
- Add sqlite graph store unit test.
- Add graph retriever unit test with multihop assertion.
- Add m3 e2e pipeline test with cited answer assertion.

5. QA loop
- Run lint/typecheck/tests and repair all regressions.
- Run milestone 3 CLI commands end-to-end.

6. Validation + git
- Conduct architecture/security/code quality validation via reviewers.
- Commit at file level with concise conventional commit messages.
