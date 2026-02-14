"""Grounded answer composition helpers."""

from autokg_rag.answer.composer import compose_grounded_answer
from autokg_rag.answer.grounding import score_sentence_support, split_sentences
from autokg_rag.answer.llm_adapter import (
    ExtractiveSentenceAdapter,
    OllamaSentenceAdapter,
    SentenceAdapter,
    generate_answer,
)
from autokg_rag.answer.ollama_adapter import OllamaAdapter, get_ollama_adapter

__all__ = [
    "ExtractiveSentenceAdapter",
    "OllamaAdapter",
    "OllamaSentenceAdapter",
    "SentenceAdapter",
    "compose_grounded_answer",
    "generate_answer",
    "get_ollama_adapter",
    "score_sentence_support",
    "split_sentences",
]
