"""Evaluation metric helpers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Protocol

from autokg_rag.schemas.provenance import Citation
from autokg_rag.schemas.records import (
    ChunkRecord,
    EvalMetricAggregateRecord,
    EvalMetricRowRecord,
    EvalQuestionRecord,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class SupportsChunkId(Protocol):
    chunk_id: str


def _unique_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _citation_chunk_ids(citations: Sequence[Citation]) -> list[str]:
    return _unique_preserve_order([citation.chunk_id for citation in citations])


def _top_k_hit_chunk_ids(hits: Sequence[SupportsChunkId], k: int) -> list[str]:
    if k < 1:
        raise ValueError("k must be >= 1.")
    top_hits = hits[:k]
    return _unique_preserve_order([hit.chunk_id for hit in top_hits])


def recall_at_k(
    *,
    gold_citations: Sequence[Citation],
    hits: Sequence[SupportsChunkId],
    k: int,
) -> float:
    """Compute chunk-level recall@k."""

    gold_ids = set(_citation_chunk_ids(gold_citations))
    if not gold_ids:
        return 0.0

    hit_ids = set(_top_k_hit_chunk_ids(hits, k))
    return float(len(gold_ids & hit_ids)) / float(len(gold_ids))


def ndcg_at_k(
    *,
    gold_citations: Sequence[Citation],
    hits: Sequence[SupportsChunkId],
    k: int,
) -> float:
    """Compute nDCG@k with binary relevance on chunk IDs."""

    gold_ids = set(_citation_chunk_ids(gold_citations))
    if not gold_ids:
        return 0.0

    hit_ids = _top_k_hit_chunk_ids(hits, k)
    if not hit_ids:
        return 0.0

    dcg = 0.0
    for rank, chunk_id in enumerate(hit_ids, start=1):
        relevance = 1.0 if chunk_id in gold_ids else 0.0
        if relevance > 0.0:
            dcg += relevance / math.log2(float(rank + 1))

    ideal_count = min(len(gold_ids), k)
    if ideal_count == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(float(rank + 1)) for rank in range(1, ideal_count + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def citation_precision(
    *,
    predicted_citations: Sequence[Citation],
    gold_citations: Sequence[Citation],
) -> float:
    """Compute precision for cited chunk IDs against gold citations."""

    predicted_ids = set(_citation_chunk_ids(predicted_citations))
    if not predicted_ids:
        return 0.0
    gold_ids = set(_citation_chunk_ids(gold_citations))
    return float(len(predicted_ids & gold_ids)) / float(len(predicted_ids))


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def _chunk_text(row: ChunkRecord | str) -> str:
    if isinstance(row, ChunkRecord):
        return row.chunk_text
    return row


def faithfulness_proxy(
    *,
    answer_text: str,
    citations: Sequence[Citation],
    chunk_lookup: Mapping[str, ChunkRecord | str],
) -> float:
    """Approximate faithfulness via lexical overlap between answer and cited evidence."""

    answer_tokens = _tokenize(answer_text)
    if not answer_tokens or not citations:
        return 0.0

    support_tokens: set[str] = set()
    for citation in citations:
        chunk = chunk_lookup.get(citation.chunk_id)
        if chunk is None:
            continue
        support_tokens.update(_tokenize(_chunk_text(chunk)))

    if not support_tokens:
        return 0.0
    return float(len(answer_tokens & support_tokens)) / float(len(answer_tokens))


def evaluate_query_metrics(
    *,
    question: EvalQuestionRecord,
    hits: Sequence[SupportsChunkId],
    k: int,
    predicted_citations: Sequence[Citation] | None = None,
    answer_text: str | None = None,
    chunk_lookup: Mapping[str, ChunkRecord | str] | None = None,
) -> EvalMetricRowRecord:
    """Compute all per-query metrics for a single question."""

    resolved_citations = [] if predicted_citations is None else list(predicted_citations)
    resolved_chunk_lookup: Mapping[str, ChunkRecord | str] = (
        {} if chunk_lookup is None else chunk_lookup
    )
    resolved_answer_text = "" if answer_text is None else answer_text

    return EvalMetricRowRecord(
        question_id=question.question_id,
        k=k,
        recall_at_k=recall_at_k(gold_citations=question.gold_citations, hits=hits, k=k),
        ndcg_at_k=ndcg_at_k(gold_citations=question.gold_citations, hits=hits, k=k),
        citation_precision=citation_precision(
            predicted_citations=resolved_citations,
            gold_citations=question.gold_citations,
        ),
        faithfulness_proxy=faithfulness_proxy(
            answer_text=resolved_answer_text,
            citations=resolved_citations,
            chunk_lookup=resolved_chunk_lookup,
        ),
    )


def aggregate_metric_rows(rows: Sequence[EvalMetricRowRecord]) -> EvalMetricAggregateRecord:
    """Average metric rows across queries."""

    query_count = len(rows)
    if query_count == 0:
        return EvalMetricAggregateRecord(
            query_count=0,
            recall_at_k=0.0,
            ndcg_at_k=0.0,
            citation_precision=0.0,
            faithfulness_proxy=0.0,
        )

    return EvalMetricAggregateRecord(
        query_count=query_count,
        recall_at_k=sum(row.recall_at_k for row in rows) / float(query_count),
        ndcg_at_k=sum(row.ndcg_at_k for row in rows) / float(query_count),
        citation_precision=sum(row.citation_precision for row in rows) / float(query_count),
        faithfulness_proxy=sum(row.faithfulness_proxy for row in rows) / float(query_count),
    )


def evaluate_and_aggregate(
    *,
    questions: Sequence[EvalQuestionRecord],
    hits_by_question_id: Mapping[str, Sequence[SupportsChunkId]],
    k: int,
    predicted_citations_by_question_id: Mapping[str, Sequence[Citation]] | None = None,
    answers_by_question_id: Mapping[str, str] | None = None,
    chunk_lookup: Mapping[str, ChunkRecord | str] | None = None,
) -> tuple[list[EvalMetricRowRecord], EvalMetricAggregateRecord]:
    """Compute per-query metric rows and aggregate summary."""

    per_query: list[EvalMetricRowRecord] = []
    citation_lookup = (
        {} if predicted_citations_by_question_id is None else predicted_citations_by_question_id
    )
    answer_lookup = {} if answers_by_question_id is None else answers_by_question_id
    resolved_chunk_lookup: Mapping[str, ChunkRecord | str] = (
        {} if chunk_lookup is None else chunk_lookup
    )

    for question in questions:
        row = evaluate_query_metrics(
            question=question,
            hits=hits_by_question_id.get(question.question_id, []),
            k=k,
            predicted_citations=citation_lookup.get(question.question_id, []),
            answer_text=answer_lookup.get(question.question_id, ""),
            chunk_lookup=resolved_chunk_lookup,
        )
        per_query.append(row)

    return per_query, aggregate_metric_rows(per_query)
