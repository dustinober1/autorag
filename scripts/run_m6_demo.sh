#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${AUTORAG_DEMO_RUN_ID:-m6}"
INPUT_DIR="${AUTORAG_DEMO_INPUT_DIR:-data/fixtures/pdfs}"
QUESTION="${AUTORAG_DEMO_QUESTION:-Compare mitigation and acceptance strategies.}"
MODE="${AUTORAG_DEMO_MODE:-hybrid}"
TOP_K="${AUTORAG_DEMO_TOP_K:-8}"
REPORTS_DIR="${AUTORAG_DEMO_REPORTS_DIR:-reports/milestones}"
MATRIX_REPORTS_DIR="${AUTORAG_DEMO_MATRIX_REPORTS_DIR:-reports/experiments}"

uv run autorag demo-build \
  --run-id "${RUN_ID}" \
  --input "${INPUT_DIR}" \
  --question "${QUESTION}" \
  --mode "${MODE}" \
  --top-k "${TOP_K}" \
  --reports-dir "${REPORTS_DIR}" \
  --matrix-reports-dir "${MATRIX_REPORTS_DIR}"
