"""Shared reranking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar


class RerankableHit(Protocol):
    """Minimal hit contract used by rerank helpers."""

    chunk_id: str

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> Any:
        """Return a copied model with optional field overrides."""


HitT = TypeVar("HitT", bound=RerankableHit)


@dataclass(frozen=True)
class RerankResult:
    """Rerank output plus parse/debug metadata."""

    hits: list[Any]
    parse_status: str
    prompt_hash: str | None = None
    raw_output: str | None = None
    error: str | None = None


def with_sequential_ranks(hits: list[HitT]) -> list[HitT]:
    """Return hits in the same order with rank values rewritten 1..N."""

    return [hit.model_copy(update={"rank": idx}) for idx, hit in enumerate(hits, start=1)]


def reorder_hits_by_chunk_id(*, hits: list[HitT], ranked_chunk_ids: list[str]) -> list[HitT]:
    """Return hits ordered by `ranked_chunk_ids` and renumbered 1..N."""

    hit_by_chunk_id = {hit.chunk_id: hit for hit in hits}
    reordered = [hit_by_chunk_id[chunk_id] for chunk_id in ranked_chunk_ids]
    return with_sequential_ranks(reordered)
