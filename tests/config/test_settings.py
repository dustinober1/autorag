from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from autokg_rag.config.loaders import load_settings


def test_load_settings_defaults_keep_local_deterministic_provider(tmp_path: Path) -> None:
    settings = load_settings(config_path=tmp_path / "missing.yaml")

    assert settings.embedding_provider == "local_hash"
    assert settings.embedding_model == "bge-small-en-v1.5"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_timeout_seconds == 30.0
    assert settings.reranker_enabled is False
    assert settings.reranker_model == "llama3:8b"
    assert settings.reranker_candidate_k == 30


def test_load_settings_parses_new_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUTORAG_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("AUTORAG_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("AUTORAG_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("AUTORAG_OLLAMA_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("AUTORAG_RERANKER_ENABLED", "true")
    monkeypatch.setenv("AUTORAG_RERANKER_MODEL", "bge-reranker-large")
    monkeypatch.setenv("AUTORAG_RERANKER_CANDIDATE_K", "64")

    settings = load_settings(config_path=tmp_path / "missing.yaml")

    assert settings.embedding_provider == "ollama"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_timeout_seconds == 12.5
    assert settings.reranker_enabled is True
    assert settings.reranker_model == "bge-reranker-large"
    assert settings.reranker_candidate_k == 64


def test_load_settings_invalid_env_types_raise_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUTORAG_RERANKER_ENABLED", "not-a-bool")

    with pytest.raises(ValidationError):
        load_settings(config_path=tmp_path / "missing.yaml")
