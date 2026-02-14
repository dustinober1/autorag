from __future__ import annotations

from collections import Counter
from pathlib import Path

from autokg_rag.eval.dataset_builder import generate_dataset_from_chunks
from autokg_rag.io import read_jsonl_rows, write_jsonl_rows


def _write_chunk_fixture(path: Path) -> None:
    rows = [
        {
            "chunk_id": "fixture_doc_a_p1_scope_c0001",
            "doc_id": "fixture_doc_a",
            "page": 1,
            "section": "Scope Control",
            "chunk_text": "Scope control governs change approval and impact review.",
        },
        {
            "chunk_id": "fixture_doc_a_p2_schedule_c0001",
            "doc_id": "fixture_doc_a",
            "page": 2,
            "section": "Schedule Planning",
            "chunk_text": "Schedule planning aligns milestones and delivery windows.",
        },
        {
            "chunk_id": "fixture_doc_b_p1_risk_c0001",
            "doc_id": "fixture_doc_b",
            "page": 1,
            "section": "Risk Response",
            "chunk_text": "Risk response updates mitigation and contingency actions.",
        },
        {
            "chunk_id": "fixture_doc_b_p2_quality_c0001",
            "doc_id": "fixture_doc_b",
            "page": 2,
            "section": "Quality Assurance",
            "chunk_text": "Quality assurance validates acceptance criteria and defects.",
        },
    ]
    write_jsonl_rows(path, rows)


def test_generated_dataset_hits_target_size_and_type_mix(tmp_path: Path) -> None:
    input_artifact_dir = tmp_path / "m4_artifacts"
    input_artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_chunk_fixture(input_artifact_dir / "chunks.jsonl")

    artifact_root = tmp_path / "artifacts"
    target_size = 200
    output_path = generate_dataset_from_chunks(
        run_id="m5",
        input_artifact_dir=input_artifact_dir,
        target_size=target_size,
        output_artifact_root=artifact_root,
    )

    assert output_path.exists()
    rows = read_jsonl_rows(output_path)
    assert len(rows) == target_size

    question_ids = {str(row["question_id"]) for row in rows}
    assert len(question_ids) == target_size

    type_counts = Counter(str(row["type"]) for row in rows)
    assert set(type_counts) == {"fact", "multi_hop", "contrast"}
    assert type_counts["fact"] == 90
    assert type_counts["multi_hop"] == 70
    assert type_counts["contrast"] == 40

    assert all(str(row["question"]).strip() for row in rows)
    assert all(row["gold_citations"] for row in rows)
