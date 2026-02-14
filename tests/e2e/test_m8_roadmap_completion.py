from __future__ import annotations

import re
import stat
import subprocess
import tomllib
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_roadmap_profiles_and_scripts_exist_and_are_valid() -> None:
    repo_root = _repo_root()

    required_chunking_configs = {
        "configs/chunking/fixed_400_50.yaml": {
            "chunk_word_size": 400,
            "chunk_word_overlap": 50,
        },
        "configs/chunking/heading_recursive.yaml": {
            "chunk_word_size": 240,
            "chunk_word_overlap": 40,
        },
        "configs/chunking/sentence_window_5.yaml": {
            "sentence_window_size": 5,
            "chunk_word_overlap": 20,
        },
        "configs/chunking/semantic_breakpoint.yaml": {
            "semantic_similarity_breakpoint": 0.35,
        },
    }
    for rel_path, expected_fields in required_chunking_configs.items():
        path = repo_root / rel_path
        assert path.exists(), rel_path
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict), rel_path
        for key, value in expected_fields.items():
            assert loaded.get(key) == value, f"{rel_path}:{key}"

    required_embedding_configs = (
        "configs/embeddings/bge_small.yaml",
        "configs/embeddings/minilm_l6.yaml",
        "configs/embeddings/e5_small.yaml",
    )
    for rel_path in required_embedding_configs:
        path = repo_root / rel_path
        assert path.exists(), rel_path
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict), rel_path
        assert loaded.get("embedding_dim") == 384

    required_scripts = (
        "scripts/bootstrap_sample_data.sh",
        "scripts/run_m4_pipeline.sh",
        "scripts/run_m5_eval.sh",
    )
    for rel_path in required_scripts:
        path = repo_root / rel_path
        assert path.exists(), rel_path
        text = path.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash")
        is_executable = bool(path.stat().st_mode & stat.S_IXUSR)
        assert is_executable, rel_path
        syntax_check = subprocess.run(
            ["bash", "-n", str(path)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert syntax_check.returncode == 0, f"{rel_path}\n{syntax_check.stderr}"


def test_roadmap_docs_make_targets_env_and_dependencies_are_complete() -> None:
    repo_root = _repo_root()

    makefile_text = (repo_root / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^\.PHONY:.*\bm4\b.*\bm5\b", makefile_text, re.MULTILINE)
    assert re.search(r"^m4:\s*$", makefile_text, re.MULTILINE)
    assert re.search(r"^m5:\s*$", makefile_text, re.MULTILINE)
    assert "\tbash scripts/run_m4_pipeline.sh" in makefile_text
    assert "\tbash scripts/run_m5_eval.sh" in makefile_text

    docs_to_keywords = {
        "docs/architecture.md": ("local-first", "Data flow", "Provenance contract"),
        "docs/schemas.md": ("Core provenance fields", "Milestone artifact reference"),
        "docs/milestones.md": ("M4 - Hybrid retrieval", "M5 - Eval harness", "make demo-build"),
    }
    for rel_path, keywords in docs_to_keywords.items():
        text = (repo_root / rel_path).read_text(encoding="utf-8")
        assert text.strip(), rel_path
        for keyword in keywords:
            assert keyword in text, f"{rel_path}:{keyword}"

    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    for row in ("| M1 |", "| M2 |", "| M3 |", "| M4 |", "| M5 |", "| M6 |"):
        assert row in readme_text
    for command in ("make m4", "make m5", "make demo-build"):
        assert command in readme_text

    env_text = (repo_root / ".env.example").read_text(encoding="utf-8")
    for variable in (
        "AUTORAG_ARTIFACT_ROOT",
        "AUTORAG_DEMO_RUN_ID",
        "AUTORAG_DEMO_INPUT_DIR",
        "AUTORAG_DEMO_QUESTION",
        "AUTORAG_DEMO_MODE",
        "AUTORAG_DEMO_TOP_K",
        "AUTORAG_DEMO_REPORTS_DIR",
        "AUTORAG_DEMO_MATRIX_REPORTS_DIR",
    ):
        assert f"{variable}=" in env_text

    pyproject_payload = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject_payload.get("project")
    assert isinstance(project, dict)
    dependencies = project.get("dependencies")
    assert isinstance(dependencies, list)
    deps_lower = {str(dep).lower() for dep in dependencies}
    for required_dep in ("pdfplumber", "pandas", "duckdb", "networkx", "fastembed"):
        assert any(dep.startswith(required_dep) for dep in deps_lower), required_dep
