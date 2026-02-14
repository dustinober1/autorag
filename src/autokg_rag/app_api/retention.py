"""Retention helper for local artifact cleanup with safe dry-run defaults."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RetentionStatus = Literal["ok", "error"]


class RetentionCandidate(BaseModel):
    """One artifact directory selected for cleanup."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class RetentionReport(BaseModel):
    """Summarized retention plan or execution result."""

    model_config = ConfigDict(extra="forbid")

    artifact_root: str = Field(min_length=1)
    status: RetentionStatus
    dry_run: bool
    keep_latest: int = Field(ge=0)
    keep_run_ids: list[str]
    discovered_runs: list[str]
    retained_runs: list[str]
    missing_keep_run_ids: list[str]
    candidates: list[RetentionCandidate]
    deleted: list[str]
    failures: list[str]


def _sorted_run_dirs(artifact_root: Path) -> list[Path]:
    run_dirs = [candidate for candidate in artifact_root.iterdir() if candidate.is_dir()]
    return sorted(run_dirs, key=lambda candidate: (-candidate.stat().st_mtime_ns, candidate.name))


def run_artifact_retention(
    *,
    artifact_root: Path,
    keep_latest: int = 3,
    keep_run_ids: set[str] | None = None,
    dry_run: bool = True,
) -> RetentionReport:
    """Plan or apply cleanup of old run directories under artifact root."""

    resolved_keep_run_ids = set(keep_run_ids or set())

    if artifact_root.exists() and not artifact_root.is_dir():
        return RetentionReport(
            artifact_root=str(artifact_root),
            status="error",
            dry_run=dry_run,
            keep_latest=keep_latest,
            keep_run_ids=sorted(resolved_keep_run_ids),
            discovered_runs=[],
            retained_runs=[],
            missing_keep_run_ids=sorted(resolved_keep_run_ids),
            candidates=[],
            deleted=[],
            failures=[f"Artifact root exists but is not a directory: {artifact_root}."],
        )

    if not artifact_root.exists():
        return RetentionReport(
            artifact_root=str(artifact_root),
            status="ok",
            dry_run=dry_run,
            keep_latest=keep_latest,
            keep_run_ids=sorted(resolved_keep_run_ids),
            discovered_runs=[],
            retained_runs=[],
            missing_keep_run_ids=sorted(resolved_keep_run_ids),
            candidates=[],
            deleted=[],
            failures=[],
        )

    run_dirs = _sorted_run_dirs(artifact_root)
    discovered_runs = [run_dir.name for run_dir in run_dirs]
    discovered_run_set = set(discovered_runs)
    keep_from_latest = {run_dir.name for run_dir in run_dirs[:keep_latest]}
    retained_run_set = resolved_keep_run_ids | keep_from_latest

    candidates: list[RetentionCandidate] = []
    for run_dir in run_dirs:
        if run_dir.name in retained_run_set:
            continue
        candidates.append(
            RetentionCandidate(
                run_id=run_dir.name,
                path=str(run_dir),
                reason="Outside keep_latest window and not explicitly retained.",
            )
        )

    deleted: list[str] = []
    failures: list[str] = []
    if not dry_run:
        for candidate in candidates:
            candidate_path = Path(candidate.path)
            try:
                shutil.rmtree(candidate_path)
                deleted.append(candidate.path)
            except OSError as exc:
                failures.append(f"{candidate.path}: {exc}")

    status: RetentionStatus = "error" if failures else "ok"
    retained_runs = sorted(discovered_run_set & retained_run_set)
    missing_keep_run_ids = sorted(resolved_keep_run_ids - discovered_run_set)
    return RetentionReport(
        artifact_root=str(artifact_root),
        status=status,
        dry_run=dry_run,
        keep_latest=keep_latest,
        keep_run_ids=sorted(resolved_keep_run_ids),
        discovered_runs=discovered_runs,
        retained_runs=retained_runs,
        missing_keep_run_ids=missing_keep_run_ids,
        candidates=candidates,
        deleted=deleted,
        failures=failures,
    )
