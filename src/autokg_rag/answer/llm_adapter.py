"""Optional adapter interface for answer sentence generation."""

from __future__ import annotations

from typing import Protocol

from autokg_rag.answer.grounding import split_sentences


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
