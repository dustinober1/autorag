from __future__ import annotations

from autokg_rag.kg.ontology_extract import extract_ontology_from_chunks
from autokg_rag.schemas.records import ChunkRecord


def test_ontology_extractor_outputs_nodes_edges_with_evidence_chunks() -> None:
    chunks = [
        ChunkRecord(
            chunk_id="doc_a-p1-c1",
            doc_id="doc_a",
            page=1,
            section="Scope",
            chunk_text="Scope control affects risk response and requires mitigation planning.",
        ),
        ChunkRecord(
            chunk_id="doc_a-p2-c1",
            doc_id="doc_a",
            page=2,
            section="Risk",
            chunk_text="Mitigation planning influences response quality.",
        ),
    ]

    nodes, edges, _mentions = extract_ontology_from_chunks(chunks)

    assert nodes
    assert edges
    assert all(edge.evidence_chunk_ids for edge in edges)
