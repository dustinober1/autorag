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


class RetrievalHitRecord(ProvenanceRecord):
    """Vector retrieval hit payload."""

    question_id: str = Field(min_length=1)
    rank: PositiveInt
    score: float


class AnswerRecord(BaseModel):
    """Answer payload with mandatory citations."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    answer_text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)
