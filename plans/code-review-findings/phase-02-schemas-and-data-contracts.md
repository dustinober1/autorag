# Phase 02 - Schemas & Data Contracts

## Scope
- `src/autokg_rag/schemas/records.py`
- `src/autokg_rag/schemas/provenance.py`
- `src/autokg_rag/schemas/api.py`
- `src/autokg_rag/schemas/__init__.py`

## Findings

### 🟡 Minor
1. **`ChunkRecord.chunk_type` is unconstrained free text**
   - File: `src/autokg_rag/schemas/records.py:51`
   - Evidence: type is `str` with only `min_length=1`, while code comments imply a controlled taxonomy (`text`, `table`, `figure`).
   - Impact: downstream logic can receive unexpected values and silently branch incorrectly.
   - Recommendation: enforce `Literal[...]` or enum for `chunk_type`.

## Checklist Status
- [x] Provenance contract (`doc_id/page/section/chunk_id`) is enforced via `ProvenanceRecord`
- [x] API payload models use `extra="forbid"`
- [x] Citation-bearing payloads require non-empty lists (`AnswerPayload.hits`, `citation_trace`; `AnswerRecord.citations`)
- [ ] Business-rule validators beyond type/shape constraints are sparse (e.g., no semantic validation for `chunk_type` taxonomy)

## Strengths
- Provenance inheritance is consistent across `ChunkRecord`, retrieval hits, and citations.
- Pydantic model configs are strict (`extra="forbid"`) across API and artifact schemas.
