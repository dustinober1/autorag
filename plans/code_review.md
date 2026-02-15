# AutoRAG Code Review Report

## Executive Summary

AutoRAG is a well-architected, local-first knowledge-graph RAG pipeline with strong provenance tracking. The codebase demonstrates good software engineering practices with Pydantic validation, typed schemas, and modular design. However, there are several areas for improvement in security, performance, and code quality.

---

## 1. Architecture & Design ✅ Strengths

- **Clean separation of concerns**: Distinct modules for ingest, chunking, embeddings, vector store, knowledge graph, retrieval, and answer composition
- **Strong provenance contract**: Every chunk and citation anchored by `doc_id`, `page`, `section`, and `chunk_id` as documented in [`docs/architecture.md`](docs/architecture.md:37)
- **Pydantic validation throughout**: All schemas use `ConfigDict(extra="forbid")` preventing schema drift
- **Milestone-driven development**: Clear milestone progression (M1-M6) with acceptance tests
- **Pluggable components**: Strategy pattern for chunking, embedding providers, and sentence adapters

---

## 2. Security Issues 🔴 Critical

### 2.1 Path Traversal Vulnerability in Run ID Validation
**File**: [`src/autokg_rag/cli.py:50-58`](src/autokg_rag/cli.py:50)

```python
def _validated_run_id(raw_run_id: str) -> str:
    run_id = raw_run_id.strip()
    if not run_id:
        raise AutoRAGError("--run-id must be non-empty.")
    if ".." in run_id or not _RUN_ID_RE.fullmatch(run_id):
        raise AutoRAGError(...)
    return run_id
```

**Issue**: While `..` is blocked, the regex allows `/` which could still enable path traversal in some contexts. The validation should be more restrictive.

**Recommendation**: Use alphanumeric with underscore/dash only:
```python
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
```

### 2.2 API Key Exposure in Logs
**File**: [`src/autokg_rag/ollama/client.py:51-52`](src/autokg_rag/ollama/client.py:51)

The API key is added to headers but could potentially be logged in error messages or debug output.

**Recommendation**: Ensure API keys are masked in any log output and error messages.

### 2.3 Arbitrary File Read via PDF Upload
**File**: [`src/autokg_rag/app_api/document_service.py:143-151`](src/autokg_rag/app_api/document_service.py:143)

The `_save_uploads_to_temp` function writes user-uploaded files directly to disk without content validation beyond filename sanitization.

**Recommendation**: Add PDF magic byte validation before processing:
```python
if not payload.startswith(b'%PDF'):
    raise IngestError("Uploaded file is not a valid PDF.")
```

### 2.4 Missing Rate Limiting for External APIs
**File**: [`src/autokg_rag/arxiv/client.py:59`](src/autokg_rag/arxiv/client.py:59)

The arXiv client sets `delay_seconds=0` which could result in rate limiting or IP bans.

**Recommendation**: Use respectful delays (3 seconds as per arXiv guidelines).

---

## 3. Performance Concerns 🟡 Medium

### 3.1 In-Memory Loading of Large Files
**File**: [`src/autokg_rag/vector/store.py:29-37`](src/autokg_rag/vector/store.py:29)

```python
def load_embeddings(artifact_dir: Path) -> npt.NDArray[np.float32]:
    matrix = np.load(path)
    return np.asarray(matrix, dtype=np.float32)
```

**Issue**: Entire embedding matrices are loaded into memory. For large document collections, this could cause memory pressure.

**Recommendation**: Consider memory-mapped numpy arrays or chunked loading for large datasets.

### 3.2 Repeated File System Operations
**File**: [`src/autokg_rag/app_api/document_service.py:60-72`](src/autokg_rag/app_api/document_service.py:60)

Multiple parquet reads for documents, pages, and chunks in separate functions without caching.

**Recommendation**: Implement a caching layer or batch loading strategy.

### 3.3 O(n²) Similarity in Semantic Chunking
**File**: [`src/autokg_rag/chunking/base.py:213-226`](src/autokg_rag/chunking/base.py:213)

The semantic breakpoint algorithm computes Jaccard similarity for each sentence pair.

**Recommendation**: For large documents, consider approximate similarity or sliding window approach.

### 3.4 Synchronous HTTP Calls Block Event Loop
**File**: [`src/autokg_rag/ollama/client.py:64-89`](src/autokg_rag/ollama/client.py:64)

Using `urllib.request` synchronously blocks the entire process during Ollama API calls.

**Recommendation**: Consider `httpx` or `aiohttp` for async HTTP calls, especially important for the Streamlit app responsiveness.

---

## 4. Code Quality Issues 🟡 Medium

### 4.1 Duplicate Code: Answer Payload Persistence
**Files**: 
- [`src/autokg_rag/cli.py:76-85`](src/autokg_rag/cli.py:76)
- [`src/autokg_rag/app_api/service.py:233-242`](src/autokg_rag/app_api/service.py:233)

The `_persist_answer_payload` function is duplicated between CLI and service modules.

**Recommendation**: Extract to a shared utility module.

### 4.2 Inconsistent Error Handling
**File**: [`app/streamlit_app.py:302-303`](app/streamlit_app.py:302)

```python
except Exception as exc:  # noqa: BLE001
    _notify_error(st, f"Create store failed: {exc}")
```

Bare `Exception` catches throughout the Streamlit app make debugging difficult.

**Recommendation**: Catch specific exceptions and log stack traces for debugging.

### 4.3 Magic Numbers Without Constants
**File**: [`src/autokg_rag/answer/composer.py:48`](src/autokg_rag/answer/composer.py:48)

```python
if len(sentence) > 280:
    sentence = f"{sentence[:277]}..."
```

**Recommendation**: Define constants for truncation limits:
```python
MAX_SENTENCE_LENGTH = 280
TRUNCATION_SUFFIX = "..."
```

### 4.4 Missing Type Hints in Several Places
**File**: [`src/autokg_rag/eval/matrix_runner.py:105-114`](src/autokg_rag/eval/matrix_runner.py:105)

```python
def _as_str_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
```

The `raw: object` type hint is too broad.

**Recommendation**: Use more specific types or `Any` with validation.

### 4.5 Complex Functions Need Decomposition
**File**: [`src/autokg_rag/cli.py:313-395`](src/autokg_rag/cli.py:313)

The `answer` command function is 82 lines with nested conditionals.

**Recommendation**: Extract helper functions for adapter configuration and answer composition.

---

## 5. Test Coverage Observations 🟢 Good

### Strengths:
- Comprehensive E2E tests for all milestones (M1-M8)
- Good contract tests for schemas and payloads
- Integration tests for Ollama reranking and document upload
- Doctor command tests with detailed validation scenarios

### Gaps:
- **Missing unit tests for chunking strategies** - only basic tests present
- **No performance benchmarks** for large document sets
- **Missing edge case tests** for empty inputs, unicode handling, malformed PDFs
- **No concurrent access tests** for the file-based locking in [`src/autokg_rag/io/artifacts.py:47-55`](src/autokg_rag/io/artifacts.py:47)

---

## 6. Specific Code Issues

### 6.1 Potential Division by Zero
**File**: [`src/autokg_rag/eval/matrix_runner.py:553`](src/autokg_rag/eval/matrix_runner.py:553)

```python
denominator = float(query_count)
```

While there's a check for `query_count == 0` at line 532, the code path is complex and could benefit from defensive programming.

### 6.2 Unbounded Dictionary Growth
**File**: [`src/autokg_rag/kg/retriever.py:118-125`](src/autokg_rag/kg/retriever.py:118)

```python
chunk_scores: dict[str, float] = {}
for traversed_edge in traversed:
    for chunk_id in traversed_edge.evidence_chunk_ids:
        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weighted_score
```

For large knowledge graphs, this could grow unbounded.

**Recommendation**: Add a maximum limit or use a bounded cache.

### 6.3 Hardcoded Fallback Values
**File**: [`src/autokg_rag/app_api/service.py:28-30`](src/autokg_rag/app_api/service.py:28)

```python
_DEFAULT_CHUNKING_STRATEGY = "heading_recursive"
_DEFAULT_DATASET_SIZE = 20
_DEFAULT_MATRIX_TOP_K = 10
```

These should be in configuration files, not hardcoded.

---

## 7. Documentation Issues

### 7.1 Missing Docstrings
Several public functions lack docstrings:
- [`src/autokg_rag/retrieval/fusion.py`](src/autokg_rag/retrieval/fusion.py) - `fuse_hybrid_hits` needs parameter documentation
- [`src/autokg_rag/answer/grounding.py`](src/autokg_rag/answer/grounding.py) - Missing module-level documentation

### 7.2 Outdated Comments
**File**: [`src/autokg_rag/chunking/base.py:82-84`](src/autokg_rag/chunking/base.py:82)

```python
chunk_type="text",  # Default to text type
section_path="",    # Will be set later in the pipeline
cross_refs=[],      # Will be populated later
```

These comments suggest incomplete initialization - consider using `None` or making fields required.

---

## 8. Recommendations Summary

### High Priority (Security)
1. Strengthen run_id validation to prevent path traversal
2. Add PDF magic byte validation for uploads
3. Mask API keys in all log output
4. Add rate limiting for arXiv API calls

### Medium Priority (Performance)
1. Implement memory-mapped embedding loading
2. Add caching for frequently accessed parquet files
3. Consider async HTTP client for Ollama calls
4. Optimize semantic chunking for large documents

### Low Priority (Code Quality)
1. Extract duplicate `_persist_answer_payload` to shared module
2. Define constants for magic numbers
3. Decompose complex CLI functions
4. Add missing type hints and docstrings
5. Move hardcoded defaults to configuration

---

## 9. Positive Observations

- **Excellent use of Pydantic** for schema validation with `extra="forbid"`
- **Good observability** with structured logging and metrics
- **Clean CLI design** with Typer and comprehensive command coverage
- **Professional Streamlit UI** with component separation
- **Thoughtful fallback behavior** in Ollama embedding provider
- **Good test structure** with E2E milestone tests

---

## Conclusion

AutoRAG is a well-designed RAG system with strong architectural foundations. The primary concerns are around security hardening (input validation, API key handling) and performance optimization for scale. The codebase would benefit from addressing the identified security issues before production deployment, followed by performance optimization for larger document collections.