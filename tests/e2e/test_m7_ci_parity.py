from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml


def _run_commands_from_workflow(workflow_path: Path) -> list[str]:
    loaded = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    jobs = loaded.get("jobs")
    assert isinstance(jobs, dict)
    quality_gate_job = jobs.get("quality-gates")
    assert isinstance(quality_gate_job, dict)
    steps = quality_gate_job.get("steps")
    assert isinstance(steps, list)

    commands: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        run = step.get("run")
        if isinstance(run, str):
            commands.append(run.strip())
    return commands


def test_verify_target_matches_ci_commands() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    makefile_text = (repo_root / "Makefile").read_text(encoding="utf-8")
    ci_workflow_path = repo_root / ".github" / "workflows" / "ci.yml"

    assert re.search(r"^verify:\s+lint\s+typecheck\s+test\s*$", makefile_text, re.MULTILINE)

    workflow_commands = _run_commands_from_workflow(ci_workflow_path)
    assert "uv sync --extra dev" in workflow_commands
    assert "make lint" in workflow_commands
    assert "make typecheck" in workflow_commands
    assert "make test" in workflow_commands

    assert workflow_commands.index("make lint") < workflow_commands.index("make typecheck")
    assert workflow_commands.index("make typecheck") < workflow_commands.index("make test")

    verify_dry_run = subprocess.run(
        ["make", "-n", "verify"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verify_dry_run.returncode == 0, verify_dry_run.stderr
    assert "ruff check src tests" in verify_dry_run.stdout
    assert "mypy src" in verify_dry_run.stdout
    assert "pytest -q" in verify_dry_run.stdout
