# Phase 08 - Answer Composition & Grounding

## Scope
- `src/autokg_rag/answer/composer.py`
- `src/autokg_rag/answer/grounding.py`
- `src/autokg_rag/answer/llm_adapter.py`
- `src/autokg_rag/answer/ollama_adapter.py`

## Findings

### 🟡 Minor
1. **Answer composition always selects the first sentence from each chunk**
   - File: `src/autokg_rag/answer/composer.py:87-105`
   - Evidence: `chunk_sentences[0]` is used directly for every hit.
   - Impact: relevant evidence later in a chunk is ignored, lowering factual quality and faithfulness in multi-sentence chunks.
   - Recommendation: choose the best-supported sentence per chunk (or let adapter score sentence candidates) before composing.

## Checklist Status
- [x] Answer max sentence guard is enforced via function parameter validation
- [x] Citation traces include support scores and provenance
- [ ] Prompt/selection robustness can improve for multi-sentence evidence extraction

## Strengths
- Citation trace schema is explicit and non-empty.
- Composer gracefully handles missing chunks and empty sentence splits via typed errors.
