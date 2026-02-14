"""Chunking strategies."""

from autokg_rag.chunking.base import SUPPORTED_CHUNKING_STRATEGIES, ChunkingStrategy, chunk_pages
from autokg_rag.chunking.fixed import chunk_page

__all__ = [
    "SUPPORTED_CHUNKING_STRATEGIES",
    "ChunkingStrategy",
    "chunk_page",
    "chunk_pages",
]
