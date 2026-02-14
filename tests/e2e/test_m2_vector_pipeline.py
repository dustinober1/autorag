from __future__ import annotations

import json
import time
from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

from autokg_rag.cli import app


def _write_sample_pdf_set(input_dir: Path, count: int) -> None:
    for idx in range(count):
        path = input_dir / f"sample_{idx:02d}.pdf"
        path.write_text(
            (
                f"Document {idx}\n"
                "Project scope baseline governs approved changes and cost/schedule alignment.\n"
                "Risk mitigation and acceptance decisions are captured in the register.\f"
                f"Appendix {idx}\n"
                "Scope decisions require explicit change control evidence.\n"
            ),
            encoding="utf-8",
        )


def test_vector_pipeline_completes_under_600_seconds_on_sample_25_pdfs(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runner = CliRunner()

    input_dir = tmp_path / "raw" / "pdfs"
    input_dir.mkdir(parents=True)
    _write_sample_pdf_set(input_dir=input_dir, count=25)

    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("AUTORAG_ARTIFACT_ROOT", str(artifact_root))

    start = time.perf_counter()

    ingest_result = runner.invoke(
        app,
        [
            "ingest",
            "--input",
            str(input_dir),
            "--run-id",
            "m2",
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
            "m2",
            "--embedding",
            "bge-small-en-v1.5",
        ],
    )
    assert index_result.exit_code == 0, index_result.output

    query_result = runner.invoke(
        app,
        [
            "query",
            "--run-id",
            "m2",
            "--mode",
            "vector",
            "--question",
            "How is scope baseline used?",
            "--top-k",
            "8",
        ],
    )
    assert query_result.exit_code == 0, query_result.output

    elapsed_seconds = time.perf_counter() - start
    assert elapsed_seconds < 600.0

    artifact_dir = artifact_root / "m2"
    assert (artifact_dir / "documents.parquet").exists()
    assert (artifact_dir / "pages.parquet").exists()
    assert (artifact_dir / "chunks.parquet").exists()
    assert (artifact_dir / "embeddings.npy").exists()
    assert (artifact_dir / "embedding_meta.parquet").exists()
    assert (artifact_dir / "vector_hits.jsonl").exists()

    hit_rows = [
        json.loads(line)
        for line in (artifact_dir / "vector_hits.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert hit_rows

    latest_hits = hit_rows[-8:]
    scores = [float(row["score"]) for row in latest_hits]
    assert scores == sorted(scores, reverse=True)

    for row in latest_hits:
        assert row["question_id"]
        assert row["chunk_id"]
        assert row["doc_id"]
        assert row["page"] >= 1
        assert row["section"]
