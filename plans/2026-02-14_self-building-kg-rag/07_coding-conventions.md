# G) CODING CONVENTIONS

## typing rules
- Use Python type hints everywhere except trivial script entrypoints.
- Enforce `mypy --strict`; disallow implicit `Any`.
- Public function signatures must be fully typed, including return types.
- Use Pydantic models for external I/O and artifact contracts; avoid untyped dict passing between modules.

## docstring format
- Use Google-style docstrings for public modules, classes, and non-trivial functions.
- Include `Args`, `Returns`, and `Raises` when applicable.
- Keep docstrings concise; implementation details belong in code comments only when necessary.

## error handling policy
- No bare `except`; catch concrete exception types.
- Raise custom domain exceptions (`IngestError`, `SchemaError`, `RetrievalError`) with context.
- CLI commands convert domain exceptions into clear stderr messages and non-zero exit codes.
- All recoverable failures emit structured log events with `run_id`, `stage`, and `error_code`.

## model/config management approach (YAML + pydantic settings)
- All runtime options live in YAML under `configs/`; no hard-coded model names in core logic.
- `Settings` model merges `configs/base.yaml` + profile YAML + env vars + CLI overrides (precedence in that order).
- Every run writes resolved config snapshot to `data/artifacts/<run_id>/resolved_config.yaml`.
- Embedding/chunking/retrieval variants are selected by config key, not code edits.

## assumptions/defaults
- Package manager: `uv` (not Poetry).
- Python target for compatibility: `3.11` via `uv` managed interpreter.
- Default retrieval stack is fully local/offline: FastEmbed CPU models + local vector index + SQLite KG store.
- External LLMs/services are optional adapters and disabled by default.
- Demo performance target: 25 PDFs end-to-end in under 10 minutes on macOS CPU.
