# Autopilot Spec: Milestone 3 (KG + Graph Retrieval)

## Objective
Implement Milestone 3 from the project plan in the existing Python codebase:
- Automated ontology extraction from `chunks.parquet`
- Canonicalized node/edge representation
- SQLite graph store persistence
- Graph retrieval mode with multihop traversal and provenance
- CLI support for `build-kg` and `query --mode graph`
- Required Milestone 3 tests and passing QA gates

## Inputs and Preconditions
- Existing Milestone 2 artifacts and commands are available.
- Core schemas include `ChunkRecord` and `RetrievalHitRecord` provenance fields.
- CLI currently supports `smoke`, `ingest`, `index-vector`, and `query --mode vector`.

## Required Outputs
1. New artifacts for `data/artifacts/<run_id>/`:
- `kg_nodes.parquet`: `{node_id, canonical_name, node_type, aliases, confidence}`
- `kg_edges.parquet`: `{edge_id, source_node_id, relation, target_node_id, weight, evidence_chunk_ids}`
- `kg.sqlite`: tables `nodes`, `edges`, `chunk_mentions`
- `graph_hits.jsonl`: retrieval hits including provenance and score/rank

2. New CLI behavior:
- `uv run autorag build-kg --run-id m3`
- `uv run autorag query --run-id m3 --mode graph --question "..." --top-k 8`

3. New tests (exact names):
- `tests/kg/test_ontology_extraction.py::test_ontology_extractor_outputs_nodes_edges_with_evidence_chunks`
- `tests/kg/test_graph_store_sqlite.py::test_sqlite_graph_roundtrip_preserves_node_and_edge_counts`
- `tests/retrieval/test_graph_retriever.py::test_graph_retriever_returns_multihop_hits_with_provenance`
- `tests/e2e/test_m3_graph_pipeline.py::test_graph_pipeline_end_to_end_returns_cited_answer`

## Technical Design
### Ontology extraction
- Deterministic, local heuristic extraction from chunk text:
  - Candidate entities via lowercase tokenization and phrase capture around relation cues.
  - Relation cues: `affects`, `influences`, `controls`, `mitigates`, `causes`, `requires`, `uses`, `depends on`.
  - Fallback extraction from sentence pairs to ensure at least one edge per informative sentence.
- Canonicalization:
  - Normalize entity names to lowercase, trim punctuation, collapse spaces.
  - Stable node IDs from SHA1 of canonical name.
  - Merge aliases into a deduped sorted list.

### Graph storage
- Persist `nodes` and `edges` both to parquet and SQLite.
- SQLite schema:
  - `nodes(node_id PRIMARY KEY, canonical_name, node_type, aliases_json, confidence)`
  - `edges(edge_id PRIMARY KEY, source_node_id, relation, target_node_id, weight, evidence_chunk_ids_json)`
  - `chunk_mentions(chunk_id, node_id, mention_count, PRIMARY KEY(chunk_id, node_id))`

### Graph retrieval
- Build query entity seeds from normalized question tokens.
- Seed selection by lexical overlap with node canonical names.
- Multihop traversal on adjacency list loaded from SQLite/parquet (depth >=2 when available).
- Convert traversed evidence chunk IDs to `RetrievalHitRecord` rows with full provenance from chunk map.
- Stable score/rank ordering (score descending).

### Answer path for graph mode
- Reuse existing answer schema (`AnswerRecord`) and citation mechanism.
- For graph queries, write `answer.json` containing non-empty answer and citations sourced from graph hits.

## Quality Gates
- `make lint`
- `make typecheck`
- `make test`
- Targeted `uv run pytest tests/e2e/test_m3_graph_pipeline.py -q`

## Constraints
- Keep local-first/offline behavior.
- Preserve existing M1/M2 tests and commands.
- No destructive git operations.
