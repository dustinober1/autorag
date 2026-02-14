"""Sentence grounding helpers for citation support scoring."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into normalized sentence strings."""

    normalized = text.strip()
    if not normalized:
        return []

    sentences = [segment.strip() for segment in _SENTENCE_BOUNDARY_RE.split(normalized)]
    return [sentence for sentence in sentences if sentence]


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def score_sentence_support(sentence: str, evidence_text: str) -> float:
    """Score lexical support of a sentence from evidence text in [0, 1]."""

    sentence_tokens = _tokens(sentence)
    if not sentence_tokens:
        return 0.0

    evidence_tokens = _tokens(evidence_text)
    if not evidence_tokens:
        return 0.0

    overlap_count = len(sentence_tokens & evidence_tokens)
    return float(overlap_count) / float(len(sentence_tokens))
