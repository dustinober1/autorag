"""Shared artifact read/write utilities for JSONL and Parquet."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from autokg_rag.exceptions import SchemaError

fcntl: ModuleType | None
_fcntl: ModuleType | None
try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    _fcntl = None
fcntl = _fcntl


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write records to a JSONL file, replacing existing content."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{json.dumps(row, ensure_ascii=True)}\n")


def append_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append JSON rows to a JSONL file with process-level locking when available."""

    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{json.dumps(row, ensure_ascii=True)}\n" for row in rows)

    if fcntl is None:  # pragma: no cover - non-POSIX fallback
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
        return

    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""

    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SchemaError(
                f"Invalid JSONL in '{path}' at line {line_no}: {exc.msg}"
            ) from exc
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
