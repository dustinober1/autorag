"""Structured logging helper for pipeline runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class StructuredLogger:
    """JSONL logger with fixed `run_id` context."""

    run_id: str
    output_path: Path

    def __post_init__(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def info(self, stage: str, event: str, **fields: Any) -> None:
        """Emit an info-level log event."""

        self._emit("info", stage=stage, event=event, fields=fields)

    def error(self, stage: str, event: str, **fields: Any) -> None:
        """Emit an error-level log event."""

        self._emit("error", stage=stage, event=event, fields=fields)

    def _emit(self, level: str, stage: str, event: str, fields: dict[str, Any]) -> None:
        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "run_id": self.run_id,
            "stage": stage,
            "event": event,
        }
        record.update(fields)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{json.dumps(record, ensure_ascii=True)}\n")
