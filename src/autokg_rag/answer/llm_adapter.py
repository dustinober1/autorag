"""Optional adapter interface for answer sentence generation."""

from __future__ import annotations

import logging
import re
from typing import Protocol

from autokg_rag.answer.grounding import split_sentences
from autokg_rag.exceptions import RetrievalError

_CITATION_RE = re.compile(r"\[(\d+)\]")
logger = logging.getLogger(__name__)


class SentenceAdapter(Protocol):
    """Protocol for pluggable sentence generation."""

    def generate_sentences(
        self,
        *,
        question: str,
        evidence_texts: list[str],
        max_sentences: int,
    ) -> list[str]:
        """Produce answer sentences grounded in provided evidence."""


class ExtractiveSentenceAdapter:
    """Default adapter that extracts existing evidence sentences."""

    def generate_sentences(
        self,
        *,
        question: str,
        evidence_texts: list[str],
        max_sentences: int,
    ) -> list[str]:
        del question

        limit = max(1, max_sentences)
        extracted: list[str] = []
        for evidence_text in evidence_texts:
            for sentence in split_sentences(evidence_text):
                cleaned = sentence.strip()
                if not cleaned:
                    continue
                extracted.append(cleaned)
                if len(extracted) >= limit:
                    return extracted
        return extracted


class OllamaSentenceAdapter:
    """Sentence adapter that rewrites grounded evidence with Ollama."""

    def __init__(
        self,
        *,
        model: str = "llama3",
        temperature: float = 0.2,
        max_tokens: int = 512,
        ollama_base_url: str = "http://localhost:11434",
        ollama_timeout_seconds: float = 60.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_base_url = ollama_base_url
        self.ollama_timeout_seconds = ollama_timeout_seconds
        self._fallback = ExtractiveSentenceAdapter()

    def generate_sentences(
        self,
        *,
        question: str,
        evidence_texts: list[str],
        max_sentences: int,
    ) -> list[str]:
        if not evidence_texts:
            return []

        from autokg_rag.answer.ollama_adapter import get_ollama_adapter

        context = "\n\n".join(
            f"[{idx}] {text}" for idx, text in enumerate(evidence_texts, start=1)
        )
        prompt = (
            f"Question: {question}\n\n"
            f"Evidence:\n{context}\n\n"
            f"Write up to {max(1, max_sentences)} short grounded sentences "
            "that answer the question. "
            "Keep citation tags like [1], [2] inline and do not invent facts."
        )

        try:
            adapter = get_ollama_adapter(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                base_url=self.ollama_base_url,
                timeout_seconds=self.ollama_timeout_seconds,
            )
            response = adapter.generate(
                prompt,
                system_prompt=(
                    "You are a grounded assistant. Use only evidence and preserve citation markers."
                ),
            )
            sentences = [
                sentence.strip()
                for sentence in split_sentences(response)
                if sentence.strip()
            ]
            return sentences[: max(1, max_sentences)] or self._fallback.generate_sentences(
                question=question,
                evidence_texts=evidence_texts,
                max_sentences=max_sentences,
            )
        except (RetrievalError, TimeoutError, OSError, ValueError) as exc:
            logger.warning("Ollama sentence generation failed; using extractive fallback: %s", exc)
            return self._fallback.generate_sentences(
                question=question,
                evidence_texts=evidence_texts,
                max_sentences=max_sentences,
            )


def generate_answer(
    *,
    question: str,
    context_chunks: list[str],
    use_local: bool = False,
    model: str = "llama3",
    temperature: float = 0.2,
    max_tokens: int = 512,
    ollama_base_url: str = "http://localhost:11434",
    ollama_timeout_seconds: float = 60.0,
) -> tuple[str, list[dict[str, object]]]:
    """Generate an answer from context chunks with optional local Ollama inference."""

    if use_local:
        from autokg_rag.answer.ollama_adapter import get_ollama_adapter

        adapter = get_ollama_adapter(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=ollama_base_url,
            timeout_seconds=ollama_timeout_seconds,
        )
        answer, citations = adapter.generate_with_context(question, context_chunks)
        if answer.strip():
            return answer, citations

    extractive = ExtractiveSentenceAdapter()
    sentences = extractive.generate_sentences(
        question=question,
        evidence_texts=context_chunks,
        max_sentences=3,
    )
    answer = " ".join(sentences).strip()

    cited_indices = {
        int(match.group(1))
        for match in _CITATION_RE.finditer(answer)
        if match.group(1).isdigit()
    }
    citations = [
        {
            "index": idx,
            "context": chunk,
        }
        for idx, chunk in enumerate(context_chunks, start=1)
        if not cited_indices or idx in cited_indices
    ]
    return answer, citations
