from __future__ import annotations

from pathlib import Path

from autokg_rag.io import write_parquet_rows
from autokg_rag.kg.retriever import retrieve_graph_hits
from autokg_rag.kg.store_sqlite import persist_graph_sqlite
from autokg_rag.schemas.records import ChunkMentionRecord, ChunkRecord, KGEdgeRecord, KGNodeRecord


def test_graph_retriever_returns_multihop_hits_with_provenance(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "m3"
    artifact_dir.mkdir(parents=True)

    chunks = [
        ChunkRecord(
            chunk_id="doc_a-p1-c1",
            doc_id="doc_a",
            page=1,
            section="Scope",
            chunk_text="Scope control affects mitigation planning.",
        ),
        ChunkRecord(
            chunk_id="doc_a-p2-c1",
            doc_id="doc_a",
            page=2,
            section="Risk",
            chunk_text="Mitigation planning influences risk response.",
        ),
    ]
    write_parquet_rows(
        artifact_dir / "chunks.parquet",
        [chunk.model_dump(mode="json") for chunk in chunks],
    )

    nodes = [
        KGNodeRecord(
            node_id="node_scope",
            canonical_name="scope control",
            node_type="concept",
            aliases=["scope control"],
            confidence=0.8,
        ),
        KGNodeRecord(
            node_id="node_mitigation",
            canonical_name="mitigation planning",
            node_type="concept",
            aliases=["mitigation planning"],
            confidence=0.8,
        ),
        KGNodeRecord(
            node_id="node_risk",
            canonical_name="risk response",
            node_type="concept",
            aliases=["risk response"],
            confidence=0.8,
        ),
    ]
    edges = [
        KGEdgeRecord(
            edge_id="edge_1",
            source_node_id="node_scope",
            relation="affects",
            target_node_id="node_mitigation",
            weight=1.0,
            evidence_chunk_ids=["doc_a-p1-c1"],
        ),
        KGEdgeRecord(
            edge_id="edge_2",
            source_node_id="node_mitigation",
            relation="influences",
            target_node_id="node_risk",
            weight=1.0,
            evidence_chunk_ids=["doc_a-p2-c1"],
        ),
    ]
    mentions = [
        ChunkMentionRecord(chunk_id="doc_a-p1-c1", node_id="node_scope", mention_count=1),
        ChunkMentionRecord(chunk_id="doc_a-p2-c1", node_id="node_risk", mention_count=1),
    ]
    persist_graph_sqlite(
        sqlite_path=artifact_dir / "kg.sqlite",
        nodes=nodes,
        edges=edges,
        chunk_mentions=mentions,
    )

    hits = retrieve_graph_hits(
        run_id="m3",
        question="How does scope control affect risk response?",
        artifact_dir=artifact_dir,
        top_k=8,
        max_depth=2,
    )

    assert hits
    assert any(hit.chunk_id == "doc_a-p2-c1" for hit in hits)
    assert all(hit.doc_id == "doc_a" for hit in hits)
    assert all(hit.page >= 1 for hit in hits)
    assert all(hit.section for hit in hits)
