from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from autokg_rag.app_api import ollama_model_service
from autokg_rag.config import Settings


def test_list_available_models_parses_ollama_tags(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    settings = Settings(artifact_root=tmp_path / "artifacts")

    class _FakeClient:
        def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
            _ = base_url
            _ = timeout_seconds

        def list_tags(self) -> dict[str, object]:
            return {
                "models": [
                    {
                        "name": "llama3.2:3b",
                        "size": 2147483648,
                        "details": {
                            "family": "llama",
                            "parameter_size": "3B",
                            "quantization_level": "Q4_K_M",
                        },
                    },
                    {
                        "name": "nomic-embed-text",
                        "size": 536870912,
                        "details": {"family": "nomic"},
                    },
                ]
            }

    monkeypatch.setattr(ollama_model_service, "OllamaClient", _FakeClient)

    models = ollama_model_service.list_available_models(settings)
    assert [model.name for model in models] == ["llama3.2:3b", "nomic-embed-text"]
    assert models[0].size_bytes == 2147483648
    assert models[1].family == "nomic"
    assert ollama_model_service.is_embedding_model_name("nomic-embed-text") is True
    assert ollama_model_service.is_embedding_model_name("llama3.2:3b") is False


def test_check_ollama_health_returns_false_on_failure(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = Settings(artifact_root=tmp_path / "artifacts")

    def _raise_failure(_settings: Settings) -> list[object]:
        raise RuntimeError("unreachable")

    monkeypatch.setattr(ollama_model_service, "list_available_models", _raise_failure)
    assert ollama_model_service.check_ollama_health(settings) is False
