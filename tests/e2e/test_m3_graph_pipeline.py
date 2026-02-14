from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

from autokg_rag.cli import app


def test_graph_pipeline_end_to_end_returns_cited_answer(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("AUTORAG_ARTIFACT_ROOT", str(artifact_root))

    input_dir = tmp_path / "raw" / "pdfs"
    input_dir.mkdir(parents=True)

    (input_dir / "graph_fixture.pdf").write_text(
        (
            "Scope Control\n"
            "Scope control affects mitigation planning and risk response.\f"
            "Risk Response\n"
            "Mitigation planning influences risk response decisions.\n"
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
            "m3",
            "--chunking",
            "heading_recursive",
        ],
    )
    assert ingest_result.exit_code == 0, ingest_result.output

    build_result = runner.invoke(app, ["build-kg", "--run-id", "m3"])
    assert build_result.exit_code == 0, build_result.output

    query_result = runner.invoke(
        app,
        [
            "query",
            "--run-id",
            "m3",
            "--mode",
            "graph",
            "--question",
            "How does scope control affect risk response?",
            "--top-k",
            "8",
        ],
    )
    assert query_result.exit_code == 0, query_result.output

    artifact_dir = artifact_root / "m3"
    answer_path = artifact_dir / "answer.json"
    graph_hits_path = artifact_dir / "graph_hits.jsonl"
    chunks_path = artifact_dir / "chunks.parquet"

    assert answer_path.exists()
    assert graph_hits_path.exists()
    assert chunks_path.exists()

    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    assert answer["answer_text"].strip()
    assert answer["citations"]

    from autokg_rag.vector.store import load_chunks

    chunk_map = {chunk.chunk_id: chunk for chunk in load_chunks(artifact_dir)}
    for citation in answer["citations"]:
        chunk = chunk_map.get(citation["chunk_id"])
        assert chunk is not None
        assert citation["doc_id"] == chunk.doc_id
        assert citation["page"] == chunk.page
        assert citation["section"] == chunk.section
