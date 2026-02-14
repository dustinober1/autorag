# AutoRAG Architecture

## Objective

AutoRAG is a local-first knowledge-graph RAG pipeline that keeps provenance attached to every retrieval hit and answer citation.

## Data flow

```text
input PDFs
  -> ingest (manifest/pages/sections)
  -> chunking (fixed | heading_recursive | sentence_window | semantic_breakpoint)
  -> embeddings + vector index
  -> ontology extraction + SQLite KG
  -> retrieval (vector | graph | hybrid)
  -> grounded answer + citation trace
  -> eval matrix + leaderboard + demo artifacts
```

## Runtime components

- `src/autokg_rag/ingest`: PDF parsing, section detection, artifact writes.
- `src/autokg_rag/chunking`: deterministic chunk strategy implementations.
- `src/autokg_rag/embeddings` and `src/autokg_rag/vector`: embedding generation and vector retrieval.
- `src/autokg_rag/kg`: ontology extraction, graph persistence, traversal retrieval.
- `src/autokg_rag/retrieval`: hybrid fusion of vector and graph scores.
- `src/autokg_rag/answer`: grounded answer composition with citation traces.
- `src/autokg_rag/eval`: dataset generation, experiment matrix execution, and reports.
- `src/autokg_rag/app_api` and `app/streamlit_app.py`: demo service layer and UI.

## Artifact boundaries

- `data/artifacts/<run_id>`: run-scoped pipeline artifacts and resolved config snapshot.
- `reports/experiments`: matrix aggregates and per-query results.
- `reports/milestones`: milestone-focused markdown summaries.

## Provenance contract

Every chunk and citation is anchored by `doc_id`, `page`, `section`, and `chunk_id`. Retrieval and answer modules are required to preserve those fields end-to-end.
