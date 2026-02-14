"""Config loader utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from autokg_rag.config.settings import Settings


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

    env_artifact_root = os.getenv("AUTORAG_ARTIFACT_ROOT")
    if env_artifact_root:
        data["artifact_root"] = env_artifact_root

    if overrides:
        data.update(overrides)

    return Settings.model_validate(data)


def write_resolved_config(settings: Settings, output_path: Path) -> None:
    """Write the resolved settings to disk for run provenance."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = settings.model_dump(mode="json")
    output_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
