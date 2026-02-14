"""Heuristic ontology extraction from chunked text."""

from __future__ import annotations

import re
from dataclasses import dataclass

from autokg_rag.kg.canonicalize import make_edge_id, make_node_id, normalize_entity_name
from autokg_rag.schemas.records import ChunkMentionRecord, ChunkRecord, KGEdgeRecord, KGNodeRecord

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

_RELATION_CUES: tuple[tuple[str, str], ...] = (
    ("affects", "affects"),
    ("influences", "influences"),
    ("controls", "controls"),
    ("mitigates", "mitigates"),
    ("causes", "causes"),
    ("requires", "requires"),
    ("uses", "uses"),
    ("depends on", "depends_on"),
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "while",
    "with",
}


@dataclass(frozen=True)
class _NodeAccumulator:
    canonical_name: str
    aliases: set[str]
    mentions: int


@dataclass(frozen=True)
class _EdgeAccumulator:
    source_node_id: str
    relation: str
    target_node_id: str
    evidence_chunk_ids: set[str]
    weight: float


def _sentence_split(text: str) -> list[str]:
    normalized = " ".join(text.split())
    parts = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(normalized) if segment.strip()]
    return parts if parts else [normalized]


def _meaningful_tokens(text: str) -> list[str]:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    return [token for token in tokens if token not in _STOPWORDS and len(token) > 1]


def _extract_entity_fragment(fragment: str, from_end: bool) -> str:
    tokens = _meaningful_tokens(fragment)
    if not tokens:
        return ""
    selected = tokens[-3:] if from_end else tokens[:3]
    return " ".join(selected)


def _relation_candidates(sentence: str) -> list[tuple[str, str, str]]:
    lowered = sentence.lower()
    pairs: list[tuple[str, str, str]] = []

    for cue, relation in _RELATION_CUES:
        if cue not in lowered:
            continue
        left, right = lowered.split(cue, maxsplit=1)
        source = _extract_entity_fragment(left, from_end=True)
        target = _extract_entity_fragment(right, from_end=False)
        if source and target and source != target:
            pairs.append((source, relation, target))

    if pairs:
        return pairs

    tokens = _meaningful_tokens(lowered)
    if len(tokens) >= 4:
        source = " ".join(tokens[:2])
        target = " ".join(tokens[-2:])
        if source != target:
            return [(source, "related_to", target)]

    return []


def _upsert_node(
    node_acc: dict[str, _NodeAccumulator],
    mention_acc: dict[tuple[str, str], int],
    *,
    name: str,
    chunk_id: str,
) -> str:
    canonical = normalize_entity_name(name)
    if not canonical:
        return ""

    node_id = make_node_id(canonical)

    existing = node_acc.get(node_id)
    if existing is None:
        node_acc[node_id] = _NodeAccumulator(
            canonical_name=canonical,
            aliases={name.strip(), canonical},
            mentions=1,
        )
    else:
        next_aliases = set(existing.aliases)
        next_aliases.add(name.strip())
        next_aliases.add(canonical)
        node_acc[node_id] = _NodeAccumulator(
            canonical_name=existing.canonical_name,
            aliases=next_aliases,
            mentions=existing.mentions + 1,
        )

    key = (chunk_id, node_id)
    mention_acc[key] = mention_acc.get(key, 0) + 1
    return node_id


def extract_ontology_from_chunks(
    chunks: list[ChunkRecord],
) -> tuple[list[KGNodeRecord], list[KGEdgeRecord], list[ChunkMentionRecord]]:
    """Extract node/edge ontology with evidence chunk references."""

    node_acc: dict[str, _NodeAccumulator] = {}
    edge_acc: dict[str, _EdgeAccumulator] = {}
    mention_acc: dict[tuple[str, str], int] = {}

    for chunk in chunks:
        for sentence in _sentence_split(chunk.chunk_text):
            for source_name, relation, target_name in _relation_candidates(sentence):
                source_node_id = _upsert_node(
                    node_acc,
                    mention_acc,
                    name=source_name,
                    chunk_id=chunk.chunk_id,
                )
                target_node_id = _upsert_node(
                    node_acc,
                    mention_acc,
                    name=target_name,
                    chunk_id=chunk.chunk_id,
                )

                if not source_node_id or not target_node_id:
                    continue

                edge_id = make_edge_id(source_node_id, relation, target_node_id)
                existing_edge = edge_acc.get(edge_id)

                if existing_edge is None:
                    edge_acc[edge_id] = _EdgeAccumulator(
                        source_node_id=source_node_id,
                        relation=relation,
                        target_node_id=target_node_id,
                        evidence_chunk_ids={chunk.chunk_id},
                        weight=1.0,
                    )
                else:
                    next_evidence = set(existing_edge.evidence_chunk_ids)
                    next_evidence.add(chunk.chunk_id)
                    edge_acc[edge_id] = _EdgeAccumulator(
                        source_node_id=existing_edge.source_node_id,
                        relation=existing_edge.relation,
                        target_node_id=existing_edge.target_node_id,
                        evidence_chunk_ids=next_evidence,
                        weight=existing_edge.weight + 1.0,
                    )

    nodes: list[KGNodeRecord] = []
    for node_id, node_payload in sorted(node_acc.items(), key=lambda item: item[1].canonical_name):
        confidence = min(1.0, 0.55 + (0.05 * node_payload.mentions))
        nodes.append(
            KGNodeRecord(
                node_id=node_id,
                canonical_name=node_payload.canonical_name,
                node_type="concept",
                aliases=sorted(alias for alias in node_payload.aliases if alias),
                confidence=confidence,
            )
        )

    edges: list[KGEdgeRecord] = []
    for edge_id, edge_payload in sorted(edge_acc.items(), key=lambda item: item[0]):
        edges.append(
            KGEdgeRecord(
                edge_id=edge_id,
                source_node_id=edge_payload.source_node_id,
                relation=edge_payload.relation,
                target_node_id=edge_payload.target_node_id,
                weight=edge_payload.weight,
                evidence_chunk_ids=sorted(edge_payload.evidence_chunk_ids),
            )
        )

    chunk_mentions = [
        ChunkMentionRecord(
            chunk_id=chunk_id,
            node_id=node_id,
            mention_count=mention_count,
        )
        for (chunk_id, node_id), mention_count in sorted(mention_acc.items())
    ]

    return nodes, edges, chunk_mentions
