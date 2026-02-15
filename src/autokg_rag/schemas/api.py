"""API-level response contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from autokg_rag.schemas.provenance import Citation
from autokg_rag.schemas.records import AnswerRecord, DocumentType, HybridHitRecord

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


class StoreInfo(BaseModel):
    """Vector store metadata for app-level management operations."""

    model_config = ConfigDict(extra="forbid")

    store_name: str = Field(min_length=1)
    doc_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    has_embeddings: bool
    created_at: datetime


class DocumentInfo(BaseModel):
    """Document-level metadata with per-store counts."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    sha256: str = Field(min_length=10)
    document_type: DocumentType = "generic"
    page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)


class IngestResult(BaseModel):
    """Result summary returned by upload/add/import flows."""

    model_config = ConfigDict(extra="forbid")

    store_name: str = Field(min_length=1)
    documents: int = Field(ge=0)
    pages: int = Field(ge=0)
    chunks: int = Field(ge=0)
    skipped_duplicates: int = Field(default=0, ge=0)


class OllamaModelInfo(BaseModel):
    """Available Ollama model metadata returned from `/api/tags`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    family: str = ""
    parameter_size: str = ""
    quantization_level: str = ""


class ArxivPaper(BaseModel):
    """Structured arXiv search result used by the app."""

    model_config = ConfigDict(extra="forbid")

    arxiv_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    pdf_url: str = Field(min_length=1)
    published: datetime


__all__ = [
    "ArxivPaper",
    "AnswerPayload",
    "AnswerRecord",
    "CitationTraceRecord",
    "DocumentInfo",
    "HybridHitRecord",
    "IngestResult",
    "OllamaModelInfo",
    "QueryMode",
    "QueryRequest",
    "StoreInfo",
]
