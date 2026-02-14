# Artifact Schemas

## Core provenance fields

All retrieval/answer artifacts use these required fields:

- `doc_id`
- `page`
- `section`
- `chunk_id`

## Milestone artifact reference

| Milestone | Artifact | Required fields |
| --- | --- | --- |
| M1 | `doc_manifest.jsonl` | `doc_id`, `source_path`, `sha256`, `total_pages` |
| M1 | `chunks.jsonl` | `chunk_id`, `doc_id`, `page`, `section`, `chunk_text` |
| M1 | `answer.json` | `question_id`, `answer_text`, `citations[]` |
| M2 | `documents.parquet` | `doc_id`, `title`, `source_path`, `sha256` |
| M2 | `pages.parquet` | `doc_id`, `page`, `section`, `text` |
| M2 | `chunks.parquet` | `ChunkRecord` contract from `src/autokg_rag/schemas/records.py` |
| M2 | `embedding_meta.parquet` | `chunk_id`, `row_idx`, `embedding_model`, `dim` |
| M2 | `vector_hits.jsonl` | `question_id`, `rank`, `score`, provenance fields |
| M3 | `kg_nodes.parquet` | `node_id`, `canonical_name`, `node_type`, `aliases`, `confidence` |
| M3 | `kg_edges.parquet` | `edge_id`, `source_node_id`, `relation`, `target_node_id`, `weight`, `evidence_chunk_ids` |
| M3 | `kg.sqlite` | `nodes`, `edges`, `chunk_mentions` tables |
| M4 | `hybrid_hits.jsonl` | `rank`, `score`, `vector_score`, `graph_score`, provenance fields |
| M4 | `answers.jsonl` | `AnswerRecord` contract from `src/autokg_rag/schemas/records.py` |
| M4 | `citation_trace.jsonl` | `answer_sentence_id`, `citation`, `support_score` |
| M5 | `questions_300.jsonl` | `question_id`, `type`, `question`, `gold_citations[]` |
| M5 | `matrix_results.csv` | `EvalRow` columns emitted by matrix runner |
| M5 | `matrix_results.json` | `run_id`, `summary`, `rows[]` |
| M5 | `per_query/*.jsonl` | `exp_id`, `question_id`, `metrics`, `hits`, `answer` |
| M6 | `demo_payload_samples.jsonl` | `question`, `answer_record` |
| M6 | `m6_demo_report.md` | run summary, config, key metrics |

## Source of truth

- Pydantic schema models: `src/autokg_rag/schemas`.
- Evaluation row definitions: `src/autokg_rag/eval`.
- API payload contract: `src/autokg_rag/schemas/api.py`.
