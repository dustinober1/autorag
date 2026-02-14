"""Experiment reporting helpers for matrix outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from autokg_rag.exceptions import AutoRAGError

DEFAULT_REPORTS_DIR = Path("reports/experiments")


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    return 0.0


def _load_rows_from_json(json_path: Path) -> tuple[str | None, list[dict[str, Any]]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AutoRAGError(f"Invalid JSON report payload: {json_path}")

    run_id = payload.get("run_id")
    rows_raw = payload.get("rows")
    if not isinstance(rows_raw, list):
        raise AutoRAGError(f"Invalid JSON report payload (rows missing): {json_path}")

    rows: list[dict[str, Any]] = []
    for row in rows_raw:
        if isinstance(row, dict):
            rows.append(dict(row))

    return str(run_id) if run_id is not None else None, rows


def _load_rows_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            _as_float(row.get("ndcg_at_10")),
            _as_float(row.get("recall_at_10")),
            -_as_float(row.get("latency_ms")),
        ),
        reverse=True,
    )


def _format_markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    header = [
        (
            "| Rank | Experiment ID | nDCG@10 | Recall@10 | Citation Precision | "
            "Faithfulness | Latency (ms) |"
        ),
        "|---:|---|---:|---:|---:|---:|---:|",
    ]

    body: list[str] = []
    for idx, row in enumerate(rows, start=1):
        body.append(
            (
                "| {rank} | {exp_id} | {ndcg:.4f} | {recall:.4f} | "
                "{precision:.4f} | {faithfulness:.4f} | {latency:.2f} |"
            ).format(
                rank=idx,
                exp_id=row.get("exp_id", ""),
                ndcg=_as_float(row.get("ndcg_at_10")),
                recall=_as_float(row.get("recall_at_10")),
                precision=_as_float(row.get("citation_precision")),
                faithfulness=_as_float(row.get("faithfulness_proxy")),
                latency=_as_float(row.get("latency_ms")),
            )
        )

    return header + body


def build_experiment_report(
    *,
    run_id: str,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> dict[str, Any]:
    """Build and write markdown leaderboard from matrix result artifacts."""

    json_path = reports_dir / "matrix_results.json"
    csv_path = reports_dir / "matrix_results.csv"

    loaded_run_id: str | None = None
    rows: list[dict[str, Any]]

    if json_path.exists():
        loaded_run_id, rows = _load_rows_from_json(json_path)
    elif csv_path.exists():
        rows = _load_rows_from_csv(csv_path)
    else:
        raise AutoRAGError(
            "Missing matrix results. Run 'autorag eval run-matrix' before reporting."
        )

    if not rows:
        raise AutoRAGError("Matrix results are empty; cannot render leaderboard report.")

    ranked = _sorted_rows(rows)
    resolved_run_id = loaded_run_id if loaded_run_id else run_id

    report_lines = [
        "# AutoRAG Experiment Leaderboard",
        "",
        f"Run ID: `{resolved_run_id}`",
        "Primary metric: `nDCG@10`",
        "",
    ]
    report_lines.extend(_format_markdown_table(ranked))
    report_lines.append("")

    leaderboard_path = reports_dir / "leaderboard.md"
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text("\n".join(report_lines), encoding="utf-8")

    best = ranked[0]
    summary = {
        "run_id": resolved_run_id,
        "leaderboard_path": str(leaderboard_path),
        "rows": len(ranked),
        "best_exp_id": best.get("exp_id"),
        "best_ndcg_at_10": _as_float(best.get("ndcg_at_10")),
    }
    return summary
