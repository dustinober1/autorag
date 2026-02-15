# Ask Mode Rules (Non-Obvious Only)

- Package name `autorag` on PyPI but imports are `autokg_rag` (src-layout in pyproject.toml)
- `docs/` contains canonical documentation, `plans/` contains design artifacts
- Three distinct layers: `app/` (Streamlit UI) → `app_api/` (service) → `autokg_rag/` (core)
- Env vars prefixed `AUTORAG_` control demo behavior (`AUTORAG_DEMO_*`)
