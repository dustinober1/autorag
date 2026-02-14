"""Artifact record schemas for Milestone 1."""

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from autokg_rag.schemas.provenance import Citation, ProvenanceRecord


class DocumentManifestRecord(BaseModel):
    """Manifest entry for one source document."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    sha256: str = Field(min_length=10)
    total_pages: PositiveInt


class ChunkRecord(ProvenanceRecord):
    """Chunk payload with required provenance."""

    chunk_text: str = Field(min_length=1)


class AnswerRecord(BaseModel):
    """Answer payload with mandatory citations."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    answer_text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)
