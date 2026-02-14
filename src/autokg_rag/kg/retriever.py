"""Graph retrieval with multihop traversal and chunk provenance."""

from __future__ import annotations

import hashlib
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from autokg_rag.exceptions import RetrievalError
from autokg_rag.kg.store_sqlite import load_graph_sqlite
from autokg_rag.schemas.records import ChunkRecord, RetrievalHitRecord
from autokg_rag.vector.store import load_chunks

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


@dataclass(frozen=True)
class _TraversedEdge:
    edge_id: str
    depth: int
    evidence_chunk_ids: list[str]
    weight: float


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def _question_id(run_id: str, question: str) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
    return f"{run_id}:g_{digest}"


def _seed_nodes(question: str, node_names: dict[str, str]) -> list[str]:
    question_tokens = _tokens(question)

    scored: list[tuple[str, int]] = []
    for node_id, canonical_name in node_names.items():
        overlap = len(question_tokens & _tokens(canonical_name))
        if overlap > 0:
            scored.append((node_id, overlap))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [node_id for node_id, _ in scored[:4]]


def retrieve_graph_hits(
    *,
    run_id: str,
    question: str,
    artifact_dir: Path,
    top_k: int,
    max_depth: int,
) -> list[RetrievalHitRecord]:
    """Retrieve graph-based hits with provenance from traversed evidence chunks."""

    sqlite_path = artifact_dir / "kg.sqlite"
    if not sqlite_path.exists():
        raise RetrievalError(f"Missing graph database: {sqlite_path}")

    nodes, edges, _mentions = load_graph_sqlite(sqlite_path)
    if not nodes or not edges:
        raise RetrievalError("Graph store is empty. Run build-kg before graph query.")

    chunks = load_chunks(artifact_dir)
    chunk_map: dict[str, ChunkRecord] = {chunk.chunk_id: chunk for chunk in chunks}
    if not chunk_map:
        raise RetrievalError("No chunks found for graph retrieval.")

    node_names = {node.node_id: node.canonical_name for node in nodes}
    adjacency: dict[str, list[tuple[str, str, list[str], float]]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source_node_id, []).append(
            (
                edge.target_node_id,
                edge.edge_id,
                edge.evidence_chunk_ids,
                edge.weight,
            )
        )

    seed_node_ids = _seed_nodes(question=question, node_names=node_names)
    if not seed_node_ids:
        seed_node_ids = [nodes[0].node_id]

    traversed: list[_TraversedEdge] = []
    visited_at_depth: dict[str, int] = {}

    queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seed_node_ids)

    while queue:
        node_id, depth = queue.popleft()
        if depth >= max_depth:
            continue

        previous_depth = visited_at_depth.get(node_id)
        if previous_depth is not None and depth >= previous_depth:
            continue
        visited_at_depth[node_id] = depth

        for target_node_id, edge_id, evidence_chunk_ids, weight in adjacency.get(node_id, []):
            next_depth = depth + 1
            traversed.append(
                _TraversedEdge(
                    edge_id=edge_id,
                    depth=next_depth,
                    evidence_chunk_ids=evidence_chunk_ids,
                    weight=weight,
                )
            )
            queue.append((target_node_id, next_depth))

    if not traversed:
        raise RetrievalError("Graph traversal did not produce evidence chunks.")

    chunk_scores: dict[str, float] = {}
    for traversed_edge in traversed:
        depth_factor = 1.0 / float(max(1, traversed_edge.depth))
        weighted_score = traversed_edge.weight * depth_factor
        for chunk_id in traversed_edge.evidence_chunk_ids:
            if chunk_id not in chunk_map:
                continue
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weighted_score

    ranked_items = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked_items:
        raise RetrievalError("Traversed graph edges had no valid chunk evidence.")

    limited_items = ranked_items[: max(1, top_k)]
    question_id = _question_id(run_id=run_id, question=question)

    hits: list[RetrievalHitRecord] = []
    for rank, (chunk_id, score) in enumerate(limited_items, start=1):
        chunk = chunk_map[chunk_id]
        hits.append(
            RetrievalHitRecord(
                question_id=question_id,
                rank=rank,
                score=float(score),
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page=chunk.page,
                section=chunk.section,
            )
        )

    return hits
