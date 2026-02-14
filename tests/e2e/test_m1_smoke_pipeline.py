from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

from autokg_rag.cli import app


def test_smoke_pipeline_returns_cited_answer(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    runner = CliRunner()
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("AUTORAG_ARTIFACT_ROOT", str(artifact_root))

    result = runner.invoke(
        app,
        [
            "smoke",
            "--input",
            "data/fixtures/pdfs",
            "--question",
            "What is project scope?",
            "--run-id",
            "m1",
        ],
    )

    assert result.exit_code == 0, result.output

    artifact_dir = artifact_root / "m1"
    answer_path = artifact_dir / "answer.json"
    chunks_path = artifact_dir / "chunks.jsonl"

    assert answer_path.exists()
    assert chunks_path.exists()

    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    chunk_rows = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunk_map = {row["chunk_id"]: row for row in chunk_rows}

    assert str(answer["answer_text"]).strip() != ""
    assert answer["citations"]

    for citation in answer["citations"]:
        chunk = chunk_map.get(citation["chunk_id"])
        assert chunk is not None
        assert citation["doc_id"] == chunk["doc_id"]
        assert citation["page"] == chunk["page"]
        assert citation["section"] == chunk["section"]
