# Ask Mode Rules (Non-Obvious Only)

- Package is named `autorag` but importable as `autokg_rag` — the src-layout maps `src/` → `autokg_rag`
- Architecture docs in `docs/` are the canonical reference; plans in `plans/` are design artifacts, not current truth
- Milestones m1–m6 are progressive pipeline stages (ingest → vector → graph → hybrid → eval → demo)
- The Streamlit app in `app/` is the demo frontend; it calls `src/autokg_rag/app_api/` — two separate layers
- `configs/` YAML files are profiles, not environment configs — they feed into `Settings` via `loaders.py`
