"""Sidebar model selection controls backed by Ollama model discovery."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from autokg_rag.app_api.ollama_model_service import is_embedding_model_name
from autokg_rag.schemas.api import OllamaModelInfo


@dataclass(frozen=True)
class ModelSelectorState:
    """Model selections emitted from sidebar."""

    answer_model: str
    embedding_model: str
    reranker_model: str
    refresh_requested: bool


def _sidebar(st: Any) -> Any:
    return getattr(st, "sidebar", st)


def _select_model(
    sidebar: Any,
    *,
    label: str,
    options: list[str],
    default_value: str,
) -> str:
    if options:
        selected = default_value if default_value in options else options[0]
        return str(sidebar.selectbox(label, options=options, index=options.index(selected)))

    return str(sidebar.text_input(label, value=default_value))


def render_model_selector(
    st: Any,
    *,
    models: Sequence[OllamaModelInfo],
    ollama_healthy: bool,
    default_answer_model: str,
    default_embedding_model: str,
    default_reranker_model: str,
) -> ModelSelectorState:
    """Render model configuration controls."""

    sidebar = _sidebar(st)
    sidebar.markdown("### Model Configuration")

    badge_class = "health-ok" if ollama_healthy else "health-warn"
    badge_text = "Ollama reachable" if ollama_healthy else "Ollama unavailable"
    sidebar.markdown(
        (
            f'<div class="health-badge {badge_class}">'
            '<span class="dot"></span>'
            f"{badge_text}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    refresh_requested = sidebar.button("Refresh Models", use_container_width=True)
    model_names = [model.name for model in models]
    embedding_models = [name for name in model_names if is_embedding_model_name(name)]
    chat_models = [name for name in model_names if not is_embedding_model_name(name)]

    answer_model = _select_model(
        sidebar,
        label="Answer Model",
        options=chat_models or model_names,
        default_value=default_answer_model,
    )
    embedding_model = _select_model(
        sidebar,
        label="Embedding Model",
        options=embedding_models or model_names,
        default_value=default_embedding_model,
    )
    reranker_model = _select_model(
        sidebar,
        label="Reranker Model",
        options=chat_models or model_names,
        default_value=default_reranker_model,
    )

    return ModelSelectorState(
        answer_model=answer_model,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        refresh_requested=bool(refresh_requested),
    )

