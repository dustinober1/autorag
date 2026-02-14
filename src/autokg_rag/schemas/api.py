"""API-level response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from autokg_rag.schemas.provenance import Citation
from autokg_rag.schemas.records import AnswerRecord, HybridHitRecord

QueryMode = Literal["vector", "graph", "hybrid"]


class QueryRequest(BaseModel):
    """Typed query request contract for app-facing endpoints."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    mode: QueryMode = "hybrid"
    top_k: PositiveInt = 8


class CitationTraceRecord(BaseModel):
    """Sentence-level grounding trace with citation provenance."""

    model_config = ConfigDict(extra="forbid")

    answer_sentence_id: str = Field(pattern=r"^s[1-9]\d*$")
    citation: Citation
    support_score: float = Field(gt=0.0)


class AnswerPayload(BaseModel):
    """Hybrid answer payload returned by CLI/API answer flow."""

    model_config = ConfigDict(extra="forbid")

    answer: AnswerRecord
    hits: list[HybridHitRecord] = Field(min_length=1)
    citation_trace: list[CitationTraceRecord] = Field(min_length=1)


__all__ = [
    "AnswerPayload",
    "AnswerRecord",
    "CitationTraceRecord",
    "HybridHitRecord",
    "QueryMode",
    "QueryRequest",
]
