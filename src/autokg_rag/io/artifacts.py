"""Shared artifact read/write utilities for JSONL and Parquet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write records to a JSONL file, replacing existing content."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{json.dumps(row, ensure_ascii=True)}\n")


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""

    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def write_parquet_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write pylist rows into a parquet file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        table = pa.Table.from_pylist(rows)
    else:
        table = pa.table({})
    pq.write_table(table, path)


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read parquet rows into a Python list of dictionaries."""

    if not path.exists():
        return []
    table = pq.read_table(path)
    pylist = table.to_pylist()
    return [dict(row) for row in pylist]
