from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from autokg_rag.io import append_jsonl_rows, read_jsonl_rows


def _append_worker(path_text: str, start: int, count: int) -> None:
    path = Path(path_text)
    for offset in range(count):
        append_jsonl_rows(path, [{"id": start + offset}])


def test_append_jsonl_rows_creates_file_and_preserves_existing_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "rows.jsonl"

    append_jsonl_rows(output_path, [{"id": 1}, {"id": 2}])
    append_jsonl_rows(output_path, [{"id": 3}])

    rows = read_jsonl_rows(output_path)
    assert [row["id"] for row in rows] == [1, 2, 3]


def test_append_jsonl_rows_is_safe_under_parallel_process_writes(tmp_path: Path) -> None:
    output_path = tmp_path / "parallel_rows.jsonl"

    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(target=_append_worker, args=(str(output_path), 0, 30)),
        ctx.Process(target=_append_worker, args=(str(output_path), 30, 30)),
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
        assert process.exitcode == 0

    rows = read_jsonl_rows(output_path)
    ids = sorted(int(row["id"]) for row in rows)
    assert len(ids) == 60
    assert ids == list(range(60))
