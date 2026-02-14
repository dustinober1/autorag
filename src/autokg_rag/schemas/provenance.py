"""Provenance schema contracts."""

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class ProvenanceRecord(BaseModel):
    """Required provenance fields for chunk-level references."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    page: PositiveInt
    section: str = Field(min_length=1)


class Citation(ProvenanceRecord):
    """Citation reference attached to an answer payload."""
