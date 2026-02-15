# Phase 13 - arXiv Integration

## Scope
- `src/autokg_rag/arxiv/client.py`
- `src/autokg_rag/arxiv/ingest.py`

## Findings

### 🟠 Major
1. **Downloaded payloads are not validated as PDFs before persisting**
   - File: `src/autokg_rag/arxiv/client.py:119-127`
   - Evidence: raw response bytes are written directly after a non-empty check.
   - Impact: HTML/error payloads can be saved with `.pdf` extension and fail later in ingest.
   - Recommendation: validate HTTP status/content-type, enforce size bounds, and verify PDF signature (`%PDF`) before write.

2. **Shared static download directory can silently reuse stale papers**
   - File: `src/autokg_rag/arxiv/ingest.py:34-36`, `src/autokg_rag/arxiv/client.py:113-116`
   - Evidence: fixed target `data/raw/pdfs/arxiv` with short-circuit on `output_path.exists()`.
   - Impact: version updates or corrupted prior downloads are skipped; ingest can run on stale artifacts.
   - Recommendation: namespace downloads by run/store and add `force_redownload` or content/hash revalidation.

## Checklist Status
- [x] Query and max-results validation are present
- [x] arXiv result normalization produces typed `ArxivPaper` records
- [ ] Downloaded file validation and freshness controls are incomplete

## Strengths
- Search-side validation avoids empty queries and invalid result limits.
- Filename sanitization prevents obvious path-injection via paper IDs.
