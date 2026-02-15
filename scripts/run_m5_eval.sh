#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${AUTORAG_M5_RUN_ID:-m5}"
SOURCE_RUN_ID="${AUTORAG_M5_SOURCE_RUN_ID:-m4}"
TARGET_SIZE="${AUTORAG_M5_TARGET_SIZE:-200}"
REPORTS_DIR="${AUTORAG_M5_REPORTS_DIR:-reports/experiments}"
ARTIFACT_ROOT="${AUTORAG_ARTIFACT_ROOT:-data/artifacts}"

"${SCRIPT_DIR}/run_m4_pipeline.sh"

cd "${REPO_ROOT}"
uv run autorag eval generate \
  --run-id "${RUN_ID}" \
  --input "${ARTIFACT_ROOT}/${SOURCE_RUN_ID}" \
  --target-size "${TARGET_SIZE}"

DATASET_PATH="${ARTIFACT_ROOT}/${RUN_ID}/questions_${TARGET_SIZE}.jsonl"
MATRIX_CONFIG="$(mktemp)"
trap 'rm -f "${MATRIX_CONFIG}"' EXIT

cat > "${MATRIX_CONFIG}" <<CONFIG
run_id: ${RUN_ID}
dataset_path: ${DATASET_PATH}
source_run_id: ${SOURCE_RUN_ID}
reports_dir: ${REPORTS_DIR}
top_k: 10
factors:
  chunking:
    - fixed_400_50
    - heading_recursive
    - sentence_window_5
    - semantic_breakpoint
  embedding:
    - BAAI/bge-small-en-v1.5
    - sentence-transformers/all-MiniLM-L6-v2
    - intfloat/e5-small-v2
  retrieval:
    - vector
    - graph
    - hybrid
CONFIG

uv run autorag eval run-matrix --run-id "${RUN_ID}" --config "${MATRIX_CONFIG}"
uv run autorag eval report --run-id "${RUN_ID}"
