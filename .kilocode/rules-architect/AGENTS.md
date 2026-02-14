# Architect Mode Rules (Non-Obvious Only)

- Pipeline is milestone-based (m1–m6): smoke → vector → graph → hybrid → eval → demo — each writes to `data/artifacts/<run_id>/`
- Config layering: YAML profile (`configs/`) → `AUTORAG_*` env vars → CLI flags — all merged in `src/autokg_rag/config/loaders.py`
- `local_hash` embedding provider is a zero-dependency deterministic stub for testing — NOT for production use
- The KG pipeline (`src/autokg_rag/kg/`) uses LLM-based ontology extraction — it requires Ollama when `embedding_provider="ollama"`
- Eval matrix (`src/autokg_rag/eval/matrix_runner.py`) runs factorial grid over chunking × embedding × retrieval strategies
- Streamlit app (`app/`) → `app_api/` service layer → core modules — three distinct layers, not two
- Plans directory (`plans/`) contains design documents; `docs/` has operational docs — keep them separate
