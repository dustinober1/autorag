"""Artifact record schemas used across milestones."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from autokg_rag.schemas.provenance import Citation, ProvenanceRecord

QuestionType = Literal["fact", "multi_hop", "contrast"]


class DocumentManifestRecord(BaseModel):
    """Manifest entry for one source document."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    sha256: str = Field(min_length=10)
    total_pages: PositiveInt


class DocumentRecord(BaseModel):
    """Document metadata for M2 artifacts."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    sha256: str = Field(min_length=10)


class PageRecord(BaseModel):
    """Parsed page-level text for M2 artifacts."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    page: PositiveInt
    section: str = Field(min_length=1)
    text: str = Field(min_length=1)


class ChunkRecord(ProvenanceRecord):
    """Chunk payload with required provenance."""

    chunk_text: str = Field(min_length=1)
    chunk_type: str = Field(default="text", min_length=1)  # e.g., "text", "table", "figure"
    section_path: str = Field(default="", min_length=0)   # Hierarchical path like "1. Introduction / 1.1 Background"
    cross_refs: list[str] = Field(default_factory=list)   # Cross-references to related chunks


class EmbeddingMetaRecord(BaseModel):
    """Metadata row for embedding matrix alignment."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    row_idx: int = Field(ge=0)
    provider: str = Field(default="local_hash", min_length=1)
    embedding_model: str = Field(min_length=1)
    dim: int = Field(ge=1)


class KGNodeRecord(BaseModel):
    """Knowledge-graph node record."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    node_type: str = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class KGEdgeRecord(BaseModel):
    """Knowledge-graph edge record with evidence provenance."""

    model_config = ConfigDict(extra="forbid")

    edge_id: str = Field(min_length=1)
    source_node_id: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    weight: float = Field(ge=0.0)
    evidence_chunk_ids: list[str] = Field(min_length=1)


class ChunkMentionRecord(BaseModel):
    """Chunk to node mention counts for graph diagnostics."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    mention_count: int = Field(ge=1)


class RetrievalHitRecord(ProvenanceRecord):
    """Retrieval hit payload with provenance."""

    question_id: str = Field(min_length=1)
    rank: PositiveInt
    score: float


class HybridHitRecord(RetrievalHitRecord):
    """Hybrid retrieval hit carrying both component scores."""

    vector_score: float
    graph_score: float


class AnswerRecord(BaseModel):
    """Answer payload with mandatory citations."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    answer_text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)


class EvalQuestionRecord(BaseModel):
    """Evaluation question with gold citations."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    type: QuestionType
    question: str = Field(min_length=1)
    gold_citations: list[Citation] = Field(default_factory=list)
    gold_answer: str | None = None


class EvalMetricRowRecord(BaseModel):
    """Per-query metric row for one `k` value."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    k: PositiveInt
    recall_at_k: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: float = Field(ge=0.0, le=1.0)
    citation_precision: float = Field(ge=0.0, le=1.0)
    faithfulness_proxy: float = Field(ge=0.0, le=1.0)


class EvalMetricAggregateRecord(BaseModel):
    """Aggregate metrics averaged across query rows."""

    model_config = ConfigDict(extra="forbid")

    query_count: int = Field(ge=0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: float = Field(ge=0.0, le=1.0)
    citation_precision: float = Field(ge=0.0, le=1.0)
    faithfulness_proxy: float = Field(ge=0.0, le=1.0)
