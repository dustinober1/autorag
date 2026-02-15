# Phase 03 - Ingestion Pipeline

## Scope
- `src/autokg_rag/ingest/pdf_parse.py`
- `src/autokg_rag/ingest/sectionize.py`
- `src/autokg_rag/ingest/pmbok_toc_parser.py`
- `src/autokg_rag/ingest/header_footer_filter.py`
- `src/autokg_rag/ingest/table_extractor.py`
- `src/autokg_rag/ingest/image_extract.py`, `image_caption.py`
- `src/autokg_rag/ingest/manifest.py`
- `src/autokg_rag/ingest/pipeline.py`

## Findings

### 🟠 Major
1. **Table-derived chunks are misclassified after chunk splitting**
   - File: `src/autokg_rag/ingest/pipeline.py:281-303`
   - Evidence: chunk type is inferred with string containment (`"TABLE:" in chunk.chunk_text`) after chunking.
   - Impact: only the first segment of a table block tends to retain the prefix; later chunks get mislabeled as `text`, degrading downstream retrieval/filtering.
   - Recommendation: carry table provenance as metadata from `PageRecord` to all generated chunks instead of inspecting chunk text.

### 🟡 Minor
2. **Cross-reference extraction loses reference kind (Figure/Table/Chapter)**
   - File: `src/autokg_rag/ingest/sectionize.py:67-80`
   - Evidence: all matched patterns are normalized to `"Section {match}"` regardless of source pattern.
   - Impact: metadata consumers cannot distinguish figure/table/chapter references.
   - Recommendation: preserve matched label in output (`Figure 3.2`, `Table 2.1`, etc.).

## Checklist Status
- [x] Pipeline has structured stage logging and metric timers/counters
- [x] Empty inputs are guarded in parse/chunk flows
- [x] Manifest + artifact writes are present
- [ ] Type/lint quality for ingest modules currently fails quality gates (see Phase 01)

## Strengths
- Ingest pipeline writes documents/pages/chunks artifacts and resolved config deterministically.
- PMBOK TOC integration is cached per-document and reused across page processing.
