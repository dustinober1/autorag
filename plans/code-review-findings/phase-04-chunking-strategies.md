# Phase 04 - Chunking Strategies

## Scope
- `src/autokg_rag/chunking/base.py`
- `src/autokg_rag/chunking/fixed.py`
- `src/autokg_rag/chunking/heading_recursive.py`
- `src/autokg_rag/chunking/sentence_window.py`
- `src/autokg_rag/chunking/semantic_breakpoint.py`

## Findings
- No new phase-local critical/major logic defects identified beyond ingestion-side metadata propagation (documented in Phase 03).

## Checklist Status
- [x] Strategies preserve provenance fields (`doc_id`, `page`, `section`, deterministic `chunk_id`)
- [x] Empty input is handled (`(empty)` fallback)
- [x] Overlap/window logic is bounded with non-zero step
- [x] Unsupported strategy raises explicit `IngestError`

## Strengths
- `chunk_pages` provides a single dispatcher enforcing supported strategies.
- Chunk IDs are deterministic and strategy-specific, aiding reproducible experiments.
