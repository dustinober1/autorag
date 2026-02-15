"""Dynamic Ollama model discovery and categorization helpers."""

from __future__ import annotations

from autokg_rag.config import Settings
from autokg_rag.exceptions import RetrievalError
from autokg_rag.ollama import OllamaClient
from autokg_rag.schemas.api import OllamaModelInfo
from autokg_rag.utils import coerce_non_negative_int

_EMBED_KEYWORDS = (
    "embed",
    "embedding",
    "bge",
    "e5",
    "gte",
    "mxbai",
    "nomic-embed",
    "all-minilm",
)


def is_embedding_model_name(name: str) -> bool:
    """Heuristic model-role classification based on model name."""

    normalized = name.strip().lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in _EMBED_KEYWORDS)


def list_available_models(settings: Settings) -> list[OllamaModelInfo]:
    """Query Ollama `/api/tags` and return structured model metadata."""

    client = OllamaClient(
        base_url=settings.ollama_base_url,
        timeout_seconds=settings.ollama_timeout_seconds,
        api_key=settings.ollama_api_key,
    )
    payload = client.list_tags()
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise RetrievalError("Ollama /api/tags response missing 'models' list.")

    models: list[OllamaModelInfo] = []
    for raw in raw_models:
        if not isinstance(raw, dict):
            continue

        name = str(raw.get("name") or raw.get("model") or "").strip()
        if not name:
            continue

        details = raw.get("details")
        details_map = details if isinstance(details, dict) else {}

        models.append(
                OllamaModelInfo(
                    name=name,
                    size_bytes=coerce_non_negative_int(raw.get("size")),
                family=str(details_map.get("family") or "").strip(),
                parameter_size=str(details_map.get("parameter_size") or "").strip(),
                quantization_level=str(details_map.get("quantization_level") or "").strip(),
            )
        )

    return sorted(models, key=lambda model: model.name.lower())


def check_ollama_health(settings: Settings) -> bool:
    """Return whether local Ollama appears reachable."""

    try:
        list_available_models(settings)
    except Exception:  # noqa: BLE001
        return False
    return True
