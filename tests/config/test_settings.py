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
    assert settings.answer_use_local is False
    assert settings.answer_model == "llama3"
    assert settings.answer_temperature == 0.2
    assert settings.answer_max_tokens == 512
    assert settings.answer_max_sentences == 6


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
    monkeypatch.setenv("AUTORAG_ANSWER_USE_LOCAL", "true")
    monkeypatch.setenv("AUTORAG_ANSWER_MODEL", "mistral")
    monkeypatch.setenv("AUTORAG_ANSWER_TEMPERATURE", "0.5")
    monkeypatch.setenv("AUTORAG_ANSWER_MAX_TOKENS", "256")
    monkeypatch.setenv("AUTORAG_ANSWER_MAX_SENTENCES", "9")

    settings = load_settings(config_path=tmp_path / "missing.yaml")

    assert settings.embedding_provider == "ollama"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_timeout_seconds == 12.5
    assert settings.reranker_enabled is True
    assert settings.reranker_model == "bge-reranker-large"
    assert settings.reranker_candidate_k == 64
    assert settings.answer_use_local is True
    assert settings.answer_model == "mistral"
    assert settings.answer_temperature == 0.5
    assert settings.answer_max_tokens == 256
    assert settings.answer_max_sentences == 9


def test_load_settings_invalid_env_types_raise_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUTORAG_RERANKER_ENABLED", "not-a-bool")

    with pytest.raises(ValidationError):
        load_settings(config_path=tmp_path / "missing.yaml")


def test_load_settings_parses_ollama_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Settings should load OLLAMA_API_KEY from environment variable."""
    monkeypatch.setenv("OLLAMA_API_KEY", "test-cloud-key")

    settings = load_settings(config_path=tmp_path / "missing.yaml")

    assert settings.ollama_api_key == "test-cloud-key"
