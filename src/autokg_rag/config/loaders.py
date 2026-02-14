"""Config loader utilities."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from autokg_rag.config.settings import Settings

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def _parse_env_bool(value: str) -> bool | str:
    lowered = value.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    return value


def _parse_env_int(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _parse_env_float(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def _load_env_overrides() -> dict[str, Any]:
    env_map: dict[str, tuple[str, Callable[[str], Any]]] = {
        "AUTORAG_ARTIFACT_ROOT": ("artifact_root", str),
        "AUTORAG_EMBEDDING_PROVIDER": ("embedding_provider", str),
        "AUTORAG_EMBEDDING_MODEL": ("embedding_model", str),
        "AUTORAG_OLLAMA_BASE_URL": ("ollama_base_url", str),
        "AUTORAG_OLLAMA_TIMEOUT_SECONDS": ("ollama_timeout_seconds", _parse_env_float),
        "AUTORAG_RERANKER_ENABLED": ("reranker_enabled", _parse_env_bool),
        "AUTORAG_RERANKER_MODEL": ("reranker_model", str),
        "AUTORAG_RERANKER_CANDIDATE_K": ("reranker_candidate_k", _parse_env_int),
    }

    resolved: dict[str, Any] = {}
    for env_name, (field_name, parser) in env_map.items():
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        resolved[field_name] = parser(raw)
    return resolved


def load_settings(
    overrides: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> Settings:
    """Load settings from YAML, environment variables, and explicit overrides.

    Precedence order is: base YAML, environment variables, explicit overrides.
    """

    path = config_path or Path("configs/base.yaml")
    data: dict[str, Any] = {}

    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data.update(loaded)

    data.update(_load_env_overrides())

    if overrides:
        data.update(overrides)

    return Settings.model_validate(data)


def write_resolved_config(settings: Settings, output_path: Path) -> None:
    """Write the resolved settings to disk for run provenance."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = settings.model_dump(mode="json")
    output_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
