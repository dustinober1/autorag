# Full Repo Code Review - Consolidated Issues

## 🔴 Critical
- None identified in this review pass.

## 🟠 Major
1. `CR-001` - Malformed YAML parse errors are not wrapped in friendly config exceptions.
   - File: `src/autokg_rag/config/loaders.py:80`
   - Phase: 01
2. `CR-002` - Strict mypy gate fails (9 errors), breaking type-quality contract.
   - Files: `src/autokg_rag/arxiv/client.py:16,53,56,58`, `src/autokg_rag/app_api/ollama_model_service.py:33,36`, `src/autokg_rag/app_api/document_service.py:95`
   - Phase: 01
3. `CR-003` - Table chunk metadata is lost after split; table chunks become `text`.
   - File: `src/autokg_rag/ingest/pipeline.py:281-303`
   - Phase: 03
4. `CR-004` - Graph retriever traverses only outgoing edges; target-seeded queries can fail.
   - File: `src/autokg_rag/kg/retriever.py:72-83,103-116`
   - Phase: 06
5. `CR-005` - Matrix experiment ID collisions can overwrite per-query artifacts.
   - File: `src/autokg_rag/eval/matrix_runner.py:278-293,679-705,620-627`
   - Phase: 09
6. `CR-006` - CLI reports hard-coded matrix output paths that can be wrong.
   - File: `src/autokg_rag/cli.py:657-661`
   - Phase: 10
7. `CR-007` - App API query flow accepts unsanitized `run_id` path segments.
   - Files: `src/autokg_rag/schemas/api.py:21-24`, `src/autokg_rag/app_api/service.py:254-259`
   - Phase: 11
8. `CR-008` - Demo report uses first matrix row instead of best-performing row.
   - File: `src/autokg_rag/app_api/service.py:366-372`
   - Phase: 11
9. `CR-009` - arXiv downloads are not validated as actual PDFs before write.
   - File: `src/autokg_rag/arxiv/client.py:119-127`
   - Phase: 13
10. `CR-010` - arXiv ingest reuses stale files from static shared directory.
   - File: `src/autokg_rag/arxiv/ingest.py:34-36`, `src/autokg_rag/arxiv/client.py:113-116`
   - Phase: 13

## 🟡 Minor
11. `CR-011` - `chunk_type` schema is unconstrained free text.
    - File: `src/autokg_rag/schemas/records.py:51`
    - Phase: 02
12. `CR-012` - Cross-reference extractor normalizes all refs as `Section`.
    - File: `src/autokg_rag/ingest/sectionize.py:67-80`
    - Phase: 03
13. `CR-013` - Answer composer always chooses sentence 1 per chunk.
    - File: `src/autokg_rag/answer/composer.py:87-105`
    - Phase: 08
14. `CR-014` - Root PMBOK test file returns bools and triggers pytest warnings.
    - File: `test_pmbok_ingestion.py:19-139`
    - Phase: 15
15. `CR-015` - Test coverage gaps for identified regression paths.
    - Files: `tests/retrieval/test_graph_retriever.py:11-100`, `tests/eval/test_matrix_runner.py:160-260`
    - Phase: 15

## 🔵 Suggestions
16. `CR-016` - Add retry/backoff options to Ollama client.
    - File: `src/autokg_rag/ollama/client.py:55-80`
    - Phase: 14
17. `CR-017` - Add shell script validation to CI.
    - File: `.github/workflows/ci.yml:27-31`
    - Phase: 16

## QA Evidence (Current Run)
- `uv run ruff check src tests` -> **failed** (37 violations)
- `uv run mypy src` -> **failed** (9 errors)
- `uv run pytest -q` -> **passed** (82 passed, 4 warnings)
