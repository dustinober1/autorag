# A) ONE-PAGE EPIC

## Problem statement
- Build a local-first RAG system over PM + arXiv PDFs that supports both semantic similarity and explicit knowledge-graph traversal.
- Guarantee grounded answers with machine-checkable provenance on every citation (`doc_id`, `page`, `section`, `chunk_id`).
- Provide an evaluation harness that can compare retrieval/answer quality across chunking, embeddings, and retrieval modes, then surface results in a demo app.

## Non-goals
- No production multi-tenant deployment, auth, or cloud infra in this portfolio version.
- No mandatory paid APIs or hosted graph/vector services; external services stay optional adapters.
- No full ontology curation UI or human-in-the-loop labeling platform beyond lightweight JSONL review.

## Architecture overview (data flow diagram in ASCII)

```text
PDF folder
  |
  v
[Ingest]
  manifest -> parse pages -> detect sections -> normalize text
  |
  v
[Chunking]
  fixed | heading-recursive | sentence-window | semantic-breakpoint
  |
  v
[Provenance-rich chunk store]
  chunks.parquet/jsonl (doc_id,page,section,chunk_id,...)
  |                               |
  |                               +--> [Ontology extraction] -> nodes/edges -> kg.sqlite
  v
[Embeddings + vector index]
  embeddings.npy + metadata.parquet
  |
Question --------------------------------------------------------------+
  |                                                                   |
  +--> vector retriever ----+                                         |
  +--> graph retriever -----+--> hybrid fusion/rerank --> answerer ---+--> cited answer JSON
                                                                   |
                                                                   +--> eval harness (Recall@n, nDCG@n, citation precision, faithfulness proxy)
                                                                   +--> Streamlit demo + reports
```

## Key design choices
- `uv` + Python project with strict `ruff`, `mypy`, `pytest`; macOS CPU-first defaults.
- Local storage: Parquet/JSONL artifacts + SQLite graph store; no required daemon services.
- Provenance is a required typed contract at schema level and enforced in tests from milestone 1.
- Answering defaults to grounded extractive synthesis; optional LLM adapter is additive and off by default.
- Observability from day 1: structured JSON logs + metric hooks (counters/timers) on each pipeline stage.
