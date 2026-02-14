# E) REPO TREE (final target)

```text
AutoRAG/
├── pyproject.toml                     # uv project, deps, scripts, tool config pointers
├── uv.lock
├── Makefile                           # m1..m6, lint, typecheck, test, demo-build
├── README.md
├── .env.example
├── .gitignore
├── ruff.toml
├── mypy.ini
├── pytest.ini
├── configs/
│   ├── base.yaml
│   ├── logging.yaml
│   ├── chunking/
│   │   ├── fixed_400_50.yaml
│   │   ├── heading_recursive.yaml
│   │   ├── sentence_window_5.yaml
│   │   └── semantic_breakpoint.yaml
│   ├── embeddings/
│   │   ├── bge_small.yaml
│   │   ├── minilm_l6.yaml
│   │   └── e5_small.yaml
│   └── experiments/
│       └── matrix.yaml
├── src/
│   └── autokg_rag/
│       ├── __init__.py
│       ├── cli.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py
│       │   └── loaders.py
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── logging.py
│       │   └── metrics.py
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── provenance.py
│       │   ├── records.py
│       │   └── api.py
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── manifest.py
│       │   ├── pdf_parse.py
│       │   ├── sectionize.py
│       │   └── pipeline.py
│       ├── chunking/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── fixed.py
│       │   ├── heading_recursive.py
│       │   ├── sentence_window.py
│       │   └── semantic_breakpoint.py
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── fastembed_provider.py
│       │   └── pipeline.py
│       ├── vector/
│       │   ├── __init__.py
│       │   ├── index.py
│       │   ├── store.py
│       │   └── retriever.py
│       ├── kg/
│       │   ├── __init__.py
│       │   ├── ontology_extract.py
│       │   ├── canonicalize.py
│       │   ├── store_sqlite.py
│       │   └── retriever.py
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── fusion.py
│       │   └── hybrid.py
│       ├── answer/
│       │   ├── __init__.py
│       │   ├── composer.py
│       │   ├── grounding.py
│       │   └── llm_adapter.py
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── dataset_builder.py
│       │   ├── metrics.py
│       │   ├── matrix_runner.py
│       │   └── report.py
│       └── app_api/
│           ├── __init__.py
│           ├── service.py
│           └── endpoints.py
├── scripts/
│   ├── bootstrap_sample_data.sh
│   ├── run_m1_smoke.sh
│   ├── run_m2_pipeline.sh
│   ├── run_m3_pipeline.sh
│   ├── run_m4_pipeline.sh
│   ├── run_m5_eval.sh
│   └── run_m6_demo.sh
├── eval/
│   └── datasets/
│       └── starter_questions_20.jsonl
├── app/
│   ├── streamlit_app.py
│   ├── components.py
│   └── styles.css
├── docs/
│   ├── architecture.md
│   ├── schemas.md
│   ├── milestones.md
│   └── runbook.md
├── reports/
│   ├── experiments/                  # generated CSV/JSON/markdown
│   └── milestones/                   # generated milestone summaries
├── data/
│   ├── .gitignore                    # keep folder, ignore contents by default
│   ├── raw/
│   ├── interim/
│   ├── artifacts/
│   └── fixtures/
│       └── pdfs/
└── tests/
    ├── contracts/
    ├── observability/
    ├── ingest/
    ├── chunking/
    ├── vector/
    ├── kg/
    ├── retrieval/
    ├── answer/
    ├── eval/
    ├── app/
    └── e2e/
```
