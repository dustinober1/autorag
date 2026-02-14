"""SQLite storage for knowledge-graph artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from autokg_rag.schemas.records import ChunkMentionRecord, KGEdgeRecord, KGNodeRecord


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create required graph tables if they do not exist."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            canonical_name TEXT NOT NULL,
            node_type TEXT NOT NULL,
            aliases_json TEXT NOT NULL,
            confidence REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS edges (
            edge_id TEXT PRIMARY KEY,
            source_node_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            target_node_id TEXT NOT NULL,
            weight REAL NOT NULL,
            evidence_chunk_ids_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunk_mentions (
            chunk_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            mention_count INTEGER NOT NULL,
            PRIMARY KEY (chunk_id, node_id)
        );
        """
    )


def persist_graph_sqlite(
    *,
    sqlite_path: Path,
    nodes: list[KGNodeRecord],
    edges: list[KGEdgeRecord],
    chunk_mentions: list[ChunkMentionRecord],
) -> None:
    """Persist nodes, edges, and chunk mentions into SQLite."""

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(sqlite_path) as connection:
        initialize_schema(connection)

        connection.execute("DELETE FROM nodes")
        connection.execute("DELETE FROM edges")
        connection.execute("DELETE FROM chunk_mentions")

        connection.executemany(
            """
            INSERT INTO nodes(node_id, canonical_name, node_type, aliases_json, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    node.node_id,
                    node.canonical_name,
                    node.node_type,
                    json.dumps(node.aliases, ensure_ascii=True),
                    node.confidence,
                )
                for node in nodes
            ],
        )

        connection.executemany(
            """
            INSERT INTO edges(
                edge_id,
                source_node_id,
                relation,
                target_node_id,
                weight,
                evidence_chunk_ids_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    edge.edge_id,
                    edge.source_node_id,
                    edge.relation,
                    edge.target_node_id,
                    edge.weight,
                    json.dumps(edge.evidence_chunk_ids, ensure_ascii=True),
                )
                for edge in edges
            ],
        )

        connection.executemany(
            """
            INSERT INTO chunk_mentions(chunk_id, node_id, mention_count)
            VALUES (?, ?, ?)
            """,
            [
                (mention.chunk_id, mention.node_id, mention.mention_count)
                for mention in chunk_mentions
            ],
        )

        connection.commit()


def load_graph_sqlite(
    sqlite_path: Path,
) -> tuple[list[KGNodeRecord], list[KGEdgeRecord], list[ChunkMentionRecord]]:
    """Load nodes, edges, and chunk mentions from SQLite."""

    with sqlite3.connect(sqlite_path) as connection:
        node_rows = connection.execute(
            "SELECT node_id, canonical_name, node_type, aliases_json, confidence FROM nodes"
        ).fetchall()
        edge_rows = connection.execute(
            """
            SELECT
                edge_id,
                source_node_id,
                relation,
                target_node_id,
                weight,
                evidence_chunk_ids_json
            FROM edges
            """
        ).fetchall()
        mention_rows = connection.execute(
            "SELECT chunk_id, node_id, mention_count FROM chunk_mentions"
        ).fetchall()

    nodes = [
        KGNodeRecord(
            node_id=row[0],
            canonical_name=row[1],
            node_type=row[2],
            aliases=list(json.loads(row[3])),
            confidence=float(row[4]),
        )
        for row in node_rows
    ]
    edges = [
        KGEdgeRecord(
            edge_id=row[0],
            source_node_id=row[1],
            relation=row[2],
            target_node_id=row[3],
            weight=float(row[4]),
            evidence_chunk_ids=list(json.loads(row[5])),
        )
        for row in edge_rows
    ]
    mentions = [
        ChunkMentionRecord(
            chunk_id=row[0],
            node_id=row[1],
            mention_count=int(row[2]),
        )
        for row in mention_rows
    ]

    return nodes, edges, mentions


def load_graph_counts(sqlite_path: Path) -> tuple[int, int, int]:
    """Load row counts for nodes, edges, and chunk_mentions."""

    with sqlite3.connect(sqlite_path) as connection:
        node_count = int(connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])
        edge_count = int(connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        mention_count = int(connection.execute("SELECT COUNT(*) FROM chunk_mentions").fetchone()[0])

    return node_count, edge_count, mention_count
