from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from autokg_rag.eval.metrics import (
    citation_precision,
    faithfulness_proxy,
    ndcg_at_k,
    recall_at_k,
)
from autokg_rag.schemas.provenance import Citation


@dataclass(frozen=True)
class _Hit:
    chunk_id: str


def _citation(*, chunk_id: str, doc_id: str) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        doc_id=doc_id,
        page=1,
        section="Fixture",
    )


def test_recall_ndcg_citation_precision_faithfulness_known_case() -> None:
    gold_citations = [
        _citation(chunk_id="chunk_a", doc_id="doc_gold_a"),
        _citation(chunk_id="chunk_b", doc_id="doc_gold_b"),
    ]
    hits = [
        _Hit(chunk_id="chunk_a"),
        _Hit(chunk_id="chunk_x"),
        _Hit(chunk_id="chunk_b"),
    ]
    predicted_citations = [
        _citation(chunk_id="chunk_a", doc_id="doc_pred_a"),
        _citation(chunk_id="chunk_x", doc_id="doc_pred_x"),
    ]

    recall_score = recall_at_k(gold_citations=gold_citations, hits=hits, k=2)
    ndcg_score = ndcg_at_k(gold_citations=gold_citations, hits=hits, k=3)
    citation_precision_score = citation_precision(
        predicted_citations=predicted_citations,
        gold_citations=gold_citations,
    )
    faithfulness_score = faithfulness_proxy(
        answer_text="scope risk unsupported claim",
        citations=predicted_citations,
        chunk_lookup={
            "chunk_a": "scope risk evidence",
            "chunk_x": "unrelated detail",
        },
    )

    expected_ndcg = (1.0 + (1.0 / math.log2(4.0))) / (1.0 + (1.0 / math.log2(3.0)))

    assert recall_score == pytest.approx(0.5, abs=1e-6)
    assert ndcg_score == pytest.approx(expected_ndcg, abs=1e-6)
    assert citation_precision_score == pytest.approx(0.5, abs=1e-6)
    assert faithfulness_score == pytest.approx(0.5, abs=1e-6)
