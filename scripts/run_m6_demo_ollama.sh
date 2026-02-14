#!/usr/bin/env bash
set -euo pipefail

export AUTORAG_DEMO_RUN_ID="${AUTORAG_DEMO_RUN_ID:-m6_ollama}"
export AUTORAG_DEMO_MATRIX_REPORTS_DIR="${AUTORAG_DEMO_MATRIX_REPORTS_DIR:-reports/experiments/ollama}"
export AUTORAG_EMBEDDING_PROVIDER="${AUTORAG_EMBEDDING_PROVIDER:-ollama}"
export AUTORAG_EMBEDDING_MODEL="${AUTORAG_EMBEDDING_MODEL:-embeddinggemma:300m}"
export AUTORAG_RERANKER_ENABLED="${AUTORAG_RERANKER_ENABLED:-true}"
export AUTORAG_RERANKER_MODEL="${AUTORAG_RERANKER_MODEL:-llama3:8b}"
export AUTORAG_RERANKER_CANDIDATE_K="${AUTORAG_RERANKER_CANDIDATE_K:-30}"
export AUTORAG_OLLAMA_BASE_URL="${AUTORAG_OLLAMA_BASE_URL:-http://localhost:11434}"
export AUTORAG_OLLAMA_TIMEOUT_SECONDS="${AUTORAG_OLLAMA_TIMEOUT_SECONDS:-30}"

bash scripts/run_m6_demo.sh

uv run autorag eval run-matrix \
  --run-id "${AUTORAG_DEMO_RUN_ID}" \
  --config configs/experiments/matrix_ollama.yaml

uv run python - <<'PY'
from pathlib import Path
import os

from autokg_rag.eval.report import build_experiment_report

run_id = os.environ["AUTORAG_DEMO_RUN_ID"]
reports_dir = Path(os.environ["AUTORAG_DEMO_MATRIX_REPORTS_DIR"])
build_experiment_report(run_id=run_id, reports_dir=reports_dir)
PY

uv run autorag doctor \
  --run-id "${AUTORAG_DEMO_RUN_ID}" \
  --input "${AUTORAG_DEMO_INPUT_DIR:-data/fixtures/pdfs}" \
  --reports-dir "${AUTORAG_DEMO_REPORTS_DIR:-reports/milestones}" \
  --matrix-reports-dir "${AUTORAG_DEMO_MATRIX_REPORTS_DIR}"
