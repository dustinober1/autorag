"""Schema contracts."""

from autokg_rag.schemas.provenance import Citation
from autokg_rag.schemas.records import (
    AnswerRecord,
    ChunkMentionRecord,
    ChunkRecord,
    DocumentManifestRecord,
    DocumentRecord,
    EmbeddingMetaRecord,
    KGEdgeRecord,
    KGNodeRecord,
    PageRecord,
    RetrievalHitRecord,
)

__all__ = [
    "AnswerRecord",
    "ChunkMentionRecord",
    "ChunkRecord",
    "Citation",
    "DocumentManifestRecord",
    "DocumentRecord",
    "EmbeddingMetaRecord",
    "KGEdgeRecord",
    "KGNodeRecord",
    "PageRecord",
    "RetrievalHitRecord",
]
