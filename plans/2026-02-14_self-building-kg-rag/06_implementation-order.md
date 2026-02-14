# F) IMPLEMENTATION ORDER (FILE-BY-FILE)

1. `pyproject.toml` — Define package metadata, `uv` scripts, and dependencies (`pydantic`, `typer`, `pdfplumber`, `numpy`, `pandas`, `duckdb`, `networkx`, `fastembed`, `streamlit`, test/lint tools).
2. `.gitignore` — Ignore `data/**`, caches, `.venv`, reports outputs, while keeping folder placeholders.
3. `ruff.toml` — Enable strict lint rules and import sorting.
4. `mypy.ini` — Turn on strict mode and per-module overrides only where unavoidable.
5. `pytest.ini` — Configure markers (`slow`, `e2e`) and default test options.
6. `Makefile` — Add `lint`, `typecheck`, `test`, `m1`..`m6`, and `demo-build` targets.
7. `README.md` — Add quickstart and milestone command table.
8. `.env.example` — Document optional API keys and runtime toggles.
9. `configs/base.yaml` — Global defaults for paths, retrieval settings, and runtime options.
10. `configs/logging.yaml` — JSON log format and level defaults.
11. `configs/chunking/fixed_400_50.yaml` — Chunk params for fixed strategy.
12. `configs/chunking/heading_recursive.yaml` — Chunk params for heading-aware recursive splitting.
13. `configs/chunking/sentence_window_5.yaml` — Chunk params for sentence window strategy.
14. `configs/chunking/semantic_breakpoint.yaml` — Chunk params for semantic-breakpoint strategy.
15. `configs/embeddings/bge_small.yaml` — Embedding model config.
16. `configs/embeddings/minilm_l6.yaml` — Embedding model config.
17. `configs/embeddings/e5_small.yaml` — Embedding model config.
18. `configs/experiments/matrix.yaml` — Full experiment matrix definition and metrics config.
19. `src/autokg_rag/__init__.py` — Package init and version export.
20. `src/autokg_rag/cli.py` — Typer CLI with milestone commands and run-id handling.
21. `src/autokg_rag/config/settings.py` — Pydantic settings models for merged YAML/env/CLI config.
22. `src/autokg_rag/config/loaders.py` — Config loader/merge utilities.
23. `src/autokg_rag/observability/logging.py` — Structured logger initialization with run context.
24. `src/autokg_rag/observability/metrics.py` — Metric emitters and timer utilities.
25. `src/autokg_rag/schemas/provenance.py` — Pydantic models enforcing provenance fields.
26. `src/autokg_rag/schemas/records.py` — Artifact schemas (documents/pages/chunks/hits/nodes/edges/answers).
27. `src/autokg_rag/schemas/api.py` — Request/response models for app service.
28. `tests/contracts/test_provenance_schema.py` — Validate strict provenance contract.
29. `tests/observability/test_logging_metrics.py` — Validate logging/metrics outputs.
30. `src/autokg_rag/ingest/manifest.py` — Source file hashing and doc manifest builder.
31. `src/autokg_rag/ingest/pdf_parse.py` — PDF page extraction with fallback parser logic.
32. `src/autokg_rag/ingest/sectionize.py` — Section detection from headings/layout.
33. `src/autokg_rag/ingest/pipeline.py` — End-to-end ingest orchestration and artifact writes.
34. `tests/ingest/test_pdf_parser.py` — Parser and sectionizer correctness.
35. `src/autokg_rag/chunking/base.py` — Chunker interface and shared helpers.
36. `src/autokg_rag/chunking/fixed.py` — Fixed token chunker.
37. `src/autokg_rag/chunking/heading_recursive.py` — Heading-aware recursive chunker.
38. `src/autokg_rag/chunking/sentence_window.py` — Sentence window chunker.
39. `src/autokg_rag/chunking/semantic_breakpoint.py` — Semantic-breakpoint chunker.
40. `tests/chunking/test_chunk_strategies.py` — Strategy contract and provenance tests.
41. `src/autokg_rag/embeddings/base.py` — Embedding provider interface.
42. `src/autokg_rag/embeddings/fastembed_provider.py` — CPU embedding backend implementation.
43. `src/autokg_rag/embeddings/pipeline.py` — Batch embedding generation and persistence.
44. `src/autokg_rag/vector/store.py` — Embedding/chunk metadata loading and storage.
45. `src/autokg_rag/vector/index.py` — Vector index build/search primitives.
46. `src/autokg_rag/vector/retriever.py` — Vector retrieval with top-k hit formatting.
47. `tests/vector/test_vector_index.py` — Vector indexing and search behavior tests.
48. `src/autokg_rag/kg/ontology_extract.py` — Entity/relation extraction from chunks.
49. `src/autokg_rag/kg/canonicalize.py` — Concept normalization and alias merging.
50. `src/autokg_rag/kg/store_sqlite.py` — SQLite schema creation and persistence.
51. `src/autokg_rag/kg/retriever.py` — Graph traversal retrieval from KG store.
52. `tests/kg/test_ontology_extraction.py` — Node/edge/evidence extraction tests.
53. `tests/kg/test_graph_store_sqlite.py` — Graph store persistence tests.
54. `tests/retrieval/test_graph_retriever.py` — Graph retrieval behavior tests.
55. `src/autokg_rag/retrieval/fusion.py` — Score fusion and dedupe logic.
56. `src/autokg_rag/retrieval/hybrid.py` — Hybrid orchestrator combining vector + graph retrievers.
57. `src/autokg_rag/answer/grounding.py` — Sentence-to-citation support scorer.
58. `src/autokg_rag/answer/composer.py` — Grounded answer assembly logic.
59. `src/autokg_rag/answer/llm_adapter.py` — Optional pluggable generator adapter.
60. `tests/retrieval/test_hybrid_retriever.py` — Hybrid scoring tests.
61. `tests/answer/test_grounded_answer.py` — Citation grounding tests.
62. `tests/contracts/test_answer_payload.py` — Answer schema contract tests.
63. `src/autokg_rag/eval/dataset_builder.py` — 20-question bootstrap + 200–500 generator.
64. `src/autokg_rag/eval/metrics.py` — Recall@n, nDCG@n, citation precision, faithfulness proxy.
65. `src/autokg_rag/eval/matrix_runner.py` — Factorial experiment runner.
66. `src/autokg_rag/eval/report.py` — CSV/JSON/Markdown report writer.
67. `eval/datasets/starter_questions_20.jsonl` — Embedded starter evaluation set.
68. `tests/eval/test_question_generator.py` — Dataset generation tests.
69. `tests/eval/test_metrics.py` — Metric unit tests with known expected values.
70. `tests/eval/test_matrix_runner.py` — Matrix completeness/output tests.
71. `src/autokg_rag/app_api/service.py` — Query service orchestrating retrieval and answer composition.
72. `src/autokg_rag/app_api/endpoints.py` — API-like callable endpoints for app integration.
73. `app/streamlit_app.py` — Streamlit demo with mode toggles and citation table.
74. `app/components.py` — Reusable UI components for answer/citation rendering.
75. `app/styles.css` — Minimal custom styling.
76. `tests/app/test_api_contract.py` — Service payload tests.
77. `tests/app/test_streamlit_smoke.py` — Streamlit smoke tests.
78. `tests/e2e/test_m1_smoke_pipeline.py` — M1 acceptance test.
79. `tests/e2e/test_m2_vector_pipeline.py` — M2 acceptance test.
80. `tests/e2e/test_m3_graph_pipeline.py` — M3 acceptance test.
81. `tests/e2e/test_m4_hybrid_qa.py` — M4 acceptance test.
82. `tests/e2e/test_m5_eval_harness.py` — M5 acceptance test.
83. `tests/e2e/test_m6_demo_workflow.py` — M6 acceptance test.
84. `scripts/bootstrap_sample_data.sh` — Fixture prep script.
85. `scripts/run_m1_smoke.sh` — Milestone runner.
86. `scripts/run_m2_pipeline.sh` — Milestone runner.
87. `scripts/run_m3_pipeline.sh` — Milestone runner.
88. `scripts/run_m4_pipeline.sh` — Milestone runner.
89. `scripts/run_m5_eval.sh` — Milestone runner.
90. `scripts/run_m6_demo.sh` — Milestone runner.
91. `docs/architecture.md` — Final architecture and component contracts.
92. `docs/schemas.md` — Artifact schema reference.
93. `docs/milestones.md` — Milestone run instructions and acceptance criteria.
94. `docs/runbook.md` — End-to-end reproducibility steps.
95. `reports/milestones/.gitkeep` — Keep report directory.
96. `reports/experiments/.gitkeep` — Keep experiment report directory.
