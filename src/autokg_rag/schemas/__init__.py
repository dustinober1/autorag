"""Schema contracts."""

from autokg_rag.schemas.provenance import Citation
from autokg_rag.schemas.records import (
    AnswerRecord,
    ChunkRecord,
    DocumentManifestRecord,
    DocumentRecord,
    EmbeddingMetaRecord,
    PageRecord,
    RetrievalHitRecord,
)

__all__ = [
    "AnswerRecord",
    "ChunkRecord",
    "Citation",
    "DocumentManifestRecord",
    "DocumentRecord",
    "EmbeddingMetaRecord",
    "PageRecord",
    "RetrievalHitRecord",
]
