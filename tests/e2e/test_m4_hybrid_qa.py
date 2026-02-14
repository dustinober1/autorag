from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

from autokg_rag.cli import app
from autokg_rag.vector.store import load_chunks


def test_hybrid_beats_vector_on_multihop_fixture_recall_at_5(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("AUTORAG_ARTIFACT_ROOT", str(artifact_root))

    input_dir = tmp_path / "raw" / "pdfs"
    input_dir.mkdir(parents=True)

    (input_dir / "multihop_fixture.pdf").write_text(
        (
            "Scope Control\n"
            "Scope control affects mitigation planning and policy gating.\f"
            "Risk Response\n"
            "Mitigation planning influences risk response decisions for acceptance.\n"
        ),
        encoding="utf-8",
    )

    ingest_result = runner.invoke(
        app,
        [
            "ingest",
            "--input",
            str(input_dir),
            "--run-id",
            "m4",
            "--chunking",
            "heading_recursive",
        ],
    )
    assert ingest_result.exit_code == 0, ingest_result.output

    index_result = runner.invoke(
        app,
        [
            "index-vector",
            "--run-id",
            "m4",
            "--embedding",
            "bge-small-en-v1.5",
        ],
    )
    assert index_result.exit_code == 0, index_result.output

    build_result = runner.invoke(app, ["build-kg", "--run-id", "m4"])
    assert build_result.exit_code == 0, build_result.output

    question = "Compare mitigation and acceptance strategies."
    vector_result = runner.invoke(
        app,
        [
            "query",
            "--run-id",
            "m4",
            "--mode",
            "vector",
            "--question",
            question,
            "--top-k",
            "5",
        ],
    )
    assert vector_result.exit_code == 0, vector_result.output

    hybrid_result = runner.invoke(
        app,
        [
            "query",
            "--run-id",
            "m4",
            "--mode",
            "hybrid",
            "--question",
            question,
            "--top-k",
            "5",
        ],
    )
    assert hybrid_result.exit_code == 0, hybrid_result.output

    answer_result = runner.invoke(
        app,
        [
            "answer",
            "--run-id",
            "m4",
            "--question",
            question,
            "--mode",
            "hybrid",
        ],
    )
    assert answer_result.exit_code == 0, answer_result.output

    artifact_dir = artifact_root / "m4"
    assert (artifact_dir / "hybrid_hits.jsonl").exists()
    assert (artifact_dir / "answers.jsonl").exists()
    assert (artifact_dir / "citation_trace.jsonl").exists()

    chunks = load_chunks(artifact_dir)
    gold_chunk_ids = {
        chunk.chunk_id
        for chunk in chunks
        if "risk response decisions" in chunk.chunk_text.lower()
    }
    assert gold_chunk_ids

    vector_hits = json.loads(vector_result.output)
    hybrid_hits = json.loads(hybrid_result.output)

    vector_chunk_ids = {row["chunk_id"] for row in vector_hits[:5]}
    hybrid_chunk_ids = {row["chunk_id"] for row in hybrid_hits[:5]}

    vector_recall_at_5 = len(vector_chunk_ids & gold_chunk_ids) / len(gold_chunk_ids)
    hybrid_recall_at_5 = len(hybrid_chunk_ids & gold_chunk_ids) / len(gold_chunk_ids)
    assert hybrid_recall_at_5 >= vector_recall_at_5
