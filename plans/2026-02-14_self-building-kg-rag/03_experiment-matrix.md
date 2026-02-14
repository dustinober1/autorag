# C) EXPERIMENT MATRIX

## Define an experiment plan comparing
- 4 chunking strategies
- 3 embedding models
- vector-only vs graph-only vs hybrid

## Factor definitions
- Chunking strategies (4):
  - `fixed_400_50`
  - `heading_recursive`
  - `sentence_window_5`
  - `semantic_breakpoint`
- Embedding models (3):
  - `BAAI/bge-small-en-v1.5`
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `intfloat/e5-small-v2`
- Retrieval modes (3):
  - `vector`
  - `graph`
  - `hybrid`

## Execution plan
- Full factorial: `4 x 3 x 3 = 36` experiment IDs.
- ID format: `exp_{chunking}_{embedding}_{retrieval}`.
- For `graph` mode, embedding is still logged for matrix consistency; an invariance check asserts near-identical results across embedding values.
- Command: `uv run autorag eval run-matrix --config configs/experiments/matrix.yaml --run-id matrix_001`.

## Metrics
For each experiment, compute and persist:
- `Recall@5`, `Recall@10`
- `nDCG@5`, `nDCG@10`
- `citation_precision`
- `faithfulness_proxy`
- `latency_ms`

## Where results are stored (CSV/JSON)
- Aggregate CSV: `reports/experiments/matrix_results.csv`
- Aggregate JSON: `reports/experiments/matrix_results.json`
- Per-query detail JSONL: `reports/experiments/per_query/{exp_id}.jsonl`
- Best-config summary Markdown: `reports/experiments/leaderboard.md`
