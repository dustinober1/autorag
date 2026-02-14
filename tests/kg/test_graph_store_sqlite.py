from __future__ import annotations

from pathlib import Path

from autokg_rag.kg.store_sqlite import load_graph_counts, persist_graph_sqlite
from autokg_rag.schemas.records import ChunkMentionRecord, KGEdgeRecord, KGNodeRecord


def test_sqlite_graph_roundtrip_preserves_node_and_edge_counts(tmp_path: Path) -> None:
    nodes = [
        KGNodeRecord(
            node_id="node_a",
            canonical_name="scope control",
            node_type="concept",
            aliases=["scope control"],
            confidence=0.8,
        ),
        KGNodeRecord(
            node_id="node_b",
            canonical_name="risk response",
            node_type="concept",
            aliases=["risk response"],
            confidence=0.8,
        ),
    ]
    edges = [
        KGEdgeRecord(
            edge_id="edge_ab",
            source_node_id="node_a",
            relation="affects",
            target_node_id="node_b",
            weight=1.0,
            evidence_chunk_ids=["doc_a-p1-c1"],
        )
    ]
    mentions = [
        ChunkMentionRecord(
            chunk_id="doc_a-p1-c1",
            node_id="node_a",
            mention_count=1,
        )
    ]

    sqlite_path = tmp_path / "kg.sqlite"
    persist_graph_sqlite(
        sqlite_path=sqlite_path,
        nodes=nodes,
        edges=edges,
        chunk_mentions=mentions,
    )

    node_count, edge_count, mention_count = load_graph_counts(sqlite_path)
    assert node_count == len(nodes)
    assert edge_count == len(edges)
    assert mention_count == len(mentions)
