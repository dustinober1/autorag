# Phase 06 - Knowledge Graph

## Scope
- `src/autokg_rag/kg/ontology_extract.py`
- `src/autokg_rag/kg/canonicalize.py`
- `src/autokg_rag/kg/store_sqlite.py`
- `src/autokg_rag/kg/pipeline.py`
- `src/autokg_rag/kg/retriever.py`

## Findings

### 🟠 Major
1. **Graph traversal only follows outgoing edges from source nodes**
   - File: `src/autokg_rag/kg/retriever.py:72-83`, `src/autokg_rag/kg/retriever.py:103-114`
   - Evidence: adjacency map keys only on `edge.source_node_id`; traversal starts from matched question tokens.
   - Impact: object-centric questions that match only target nodes can fail with `RetrievalError("Graph traversal did not produce evidence chunks.")` despite valid graph evidence.
   - Recommendation: include reverse adjacency (or configurable bidirectional traversal) so target-node seeds can reach supporting edges.

## Checklist Status
- [x] SQL persistence/retrieval path exists and is exercised by tests
- [x] Graph retrieval is depth-bounded (`max_depth`)
- [ ] Retrieval robustness is incomplete for target-only node matches

## Strengths
- Retrieval hit construction preserves provenance and weighted scoring.
- Existing test `tests/retrieval/test_graph_retriever.py` validates a multihop happy path.
