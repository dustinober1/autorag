"""Grounded answer composition with sentence-level citations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from autokg_rag.answer.grounding import score_sentence_support, split_sentences
from autokg_rag.answer.llm_adapter import SentenceAdapter
from autokg_rag.exceptions import RetrievalError
from autokg_rag.schemas.api import CitationTraceRecord
from autokg_rag.schemas.provenance import Citation
from autokg_rag.schemas.records import AnswerRecord, ChunkRecord


class HybridHitLike(Protocol):
    """Minimal retrieval-hit contract used by grounded composition."""

    question_id: str
    score: float
    chunk_id: str
    doc_id: str
    page: int
    section: str


@dataclass
class _GroundedSentence:
    sentence: str
    citation: Citation
    chunk_text: str
    hit_score: float


def _normalize_sentence(sentence: str) -> str:
    cleaned = sentence.strip()
    if not cleaned:
        return ""
    if cleaned.endswith((".", "!", "?")):
        return cleaned
    return f"{cleaned}."


def _coerce_non_negative(value: float) -> float:
    return value if value >= 0.0 else 0.0


def _dedupe_hits_by_chunk(hits: Sequence[HybridHitLike]) -> list[HybridHitLike]:
    deduped: list[HybridHitLike] = []
    seen_chunk_ids: set[str] = set()
    for hit in hits:
        if hit.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(hit.chunk_id)
        deduped.append(hit)
    return deduped


def _support_score(
    *,
    sentence: str,
    chunk_text: str,
    hit_score: float,
    lexical_weight: float,
    hit_weight: float,
    floor: float,
) -> float:
    lexical_support = score_sentence_support(sentence=sentence, evidence_text=chunk_text)
    normalized_hit = hit_score / (1.0 + hit_score) if hit_score > 0.0 else 0.0
    return max(floor, (lexical_weight * lexical_support) + (hit_weight * normalized_hit))


def compose_grounded_answer(
    *,
    question: str,
    hits: Sequence[HybridHitLike],
    chunk_by_id: Mapping[str, ChunkRecord],
    max_sentences: int = 6,
    sentence_adapter: SentenceAdapter | None = None,
    support_lexical_weight: float = 0.7,
    support_hit_weight: float = 0.3,
    support_floor: float = 0.001,
) -> tuple[AnswerRecord, list[CitationTraceRecord]]:
    """Compose an answer where every output sentence maps to a citation."""

    if max_sentences < 1:
        raise ValueError("max_sentences must be >= 1")
    if support_lexical_weight < 0.0:
        raise ValueError("support_lexical_weight must be >= 0")
    if support_hit_weight < 0.0:
        raise ValueError("support_hit_weight must be >= 0")
    if support_floor < 0.0:
        raise ValueError("support_floor must be >= 0")
    if not hits:
        raise RetrievalError("Hybrid retrieval returned no hits for answer composition.")

    grounded: list[_GroundedSentence] = []
    for hit in _dedupe_hits_by_chunk(hits):
        chunk = chunk_by_id.get(hit.chunk_id)
        if chunk is None:
            continue

        chunk_sentences = split_sentences(chunk.chunk_text)
        if not chunk_sentences:
            continue

        grounded.append(
            _GroundedSentence(
                sentence=_normalize_sentence(chunk_sentences[0]),
                citation=Citation(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    page=hit.page,
                    section=hit.section,
                ),
                chunk_text=chunk.chunk_text,
                hit_score=_coerce_non_negative(hit.score),
            )
        )
        if len(grounded) >= max_sentences:
            break

    if not grounded:
        raise RetrievalError("Unable to compose answer: no hit had a resolvable source chunk.")

    if sentence_adapter is not None:
        rewritten = sentence_adapter.generate_sentences(
            question=question,
            evidence_texts=[item.chunk_text for item in grounded],
            max_sentences=len(grounded),
        )
        for idx, sentence in enumerate(rewritten[: len(grounded)]):
            normalized = _normalize_sentence(sentence)
            if normalized:
                grounded[idx].sentence = normalized

    answer_sentences = [item.sentence for item in grounded if item.sentence]
    if not answer_sentences:
        raise RetrievalError("Unable to compose answer: no grounded sentences were generated.")

    citations_by_chunk: dict[str, Citation] = {}
    for item in grounded:
        citations_by_chunk.setdefault(item.citation.chunk_id, item.citation)

    answer = AnswerRecord(
        question_id=hits[0].question_id,
        answer_text=" ".join(answer_sentences).strip(),
        citations=list(citations_by_chunk.values()),
    )

    citation_trace = [
        CitationTraceRecord(
            answer_sentence_id=f"s{idx}",
            citation=item.citation,
            support_score=round(
                _support_score(
                    sentence=item.sentence,
                    chunk_text=item.chunk_text,
                    hit_score=item.hit_score,
                    lexical_weight=support_lexical_weight,
                    hit_weight=support_hit_weight,
                    floor=support_floor,
                ),
                6,
            ),
        )
        for idx, item in enumerate(grounded, start=1)
    ]

    return answer, citation_trace
