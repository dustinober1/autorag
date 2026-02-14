from __future__ import annotations

import os
from pathlib import Path

from autokg_rag.app_api.retention import run_artifact_retention


def _create_run_dir(*, artifact_root: Path, run_id: str, mtime_seconds: int) -> Path:
    run_dir = artifact_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "marker.txt"
    marker.write_text(run_id, encoding="utf-8")
    os.utime(marker, (mtime_seconds, mtime_seconds))
    os.utime(run_dir, (mtime_seconds, mtime_seconds))
    return run_dir


def test_retention_dry_run_is_non_destructive(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _create_run_dir(artifact_root=artifact_root, run_id="run_old", mtime_seconds=10)
    _create_run_dir(artifact_root=artifact_root, run_id="run_keep", mtime_seconds=20)
    _create_run_dir(artifact_root=artifact_root, run_id="run_latest", mtime_seconds=30)

    report = run_artifact_retention(
        artifact_root=artifact_root,
        keep_latest=1,
        keep_run_ids={"run_keep"},
        dry_run=True,
    )

    assert report.status == "ok"
    assert report.dry_run is True
    assert report.retained_runs == ["run_keep", "run_latest"]
    assert [candidate.run_id for candidate in report.candidates] == ["run_old"]
    assert not report.deleted
    assert (artifact_root / "run_old").exists()


def test_retention_apply_deletes_only_candidates(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _create_run_dir(artifact_root=artifact_root, run_id="run_old", mtime_seconds=10)
    run_keep = _create_run_dir(artifact_root=artifact_root, run_id="run_keep", mtime_seconds=20)

    report = run_artifact_retention(
        artifact_root=artifact_root,
        keep_latest=0,
        keep_run_ids={"run_keep"},
        dry_run=False,
    )

    assert report.status == "ok"
    assert report.dry_run is False
    assert [candidate.run_id for candidate in report.candidates] == ["run_old"]
    assert len(report.deleted) == 1
    assert not report.failures
    assert not (artifact_root / "run_old").exists()
    assert run_keep.exists()
