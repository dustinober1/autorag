"""Metrics emission utilities."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


class MetricsWriter:
    """Writes stage metrics to a JSONL artifact file."""

    def __init__(self, run_id: str, output_path: Path) -> None:
        self._run_id = run_id
        self._output_path = output_path
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def counter(self, stage: str, metric_name: str, value: float = 1.0) -> None:
        """Write a counter-like metric event."""

        payload = {
            "run_id": self._run_id,
            "stage": stage,
            "metric_name": metric_name,
            "value": value,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with self._output_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{json.dumps(payload, ensure_ascii=True)}\n")

    @contextmanager
    def timer(self, stage: str, metric_name: str) -> Iterator[None]:
        """Measure and emit elapsed seconds for a stage."""

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.counter(stage=stage, metric_name=metric_name, value=elapsed)
