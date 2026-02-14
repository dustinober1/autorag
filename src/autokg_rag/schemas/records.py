"""Artifact record schemas used across milestones."""

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from autokg_rag.schemas.provenance import Citation, ProvenanceRecord


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


class EmbeddingMetaRecord(BaseModel):
    """Metadata row for embedding matrix alignment."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    row_idx: int = Field(ge=0)
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


class AnswerRecord(BaseModel):
    """Answer payload with mandatory citations."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    answer_text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)
