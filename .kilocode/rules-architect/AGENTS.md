# Architect Mode Rules (Non-Obvious Only)

- Three-layer architecture: `app/` → `app_api/` → core modules (don't conflate)
- KG pipeline requires Ollama for `embedding_provider="ollama"` (local_hash won't work)
- `plans/` vs `docs/` separation: design artifacts vs canonical docs
- Config layering: YAML → `AUTORAG_` env vars → CLI flags (priority order)
