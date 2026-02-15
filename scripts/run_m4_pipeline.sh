#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${AUTORAG_M4_RUN_ID:-m4}"
INPUT_DIR="${AUTORAG_M4_INPUT_DIR:-${REPO_ROOT}/data/fixtures/pdfs}"
QUESTION="${AUTORAG_M4_QUESTION:-Compare mitigation and acceptance strategies.}"
TOP_K="${AUTORAG_M4_TOP_K:-8}"
CHUNKING="${AUTORAG_M4_CHUNKING:-heading_recursive}"
EMBEDDING_MODEL="${AUTORAG_M4_EMBEDDING_MODEL:-bge-small-en-v1.5}"

"${SCRIPT_DIR}/bootstrap_sample_data.sh"

cd "${REPO_ROOT}"
uv run autorag ingest --input "${INPUT_DIR}" --run-id "${RUN_ID}" --chunking "${CHUNKING}"
uv run autorag index-vector --run-id "${RUN_ID}" --embedding "${EMBEDDING_MODEL}"
uv run autorag build-kg --run-id "${RUN_ID}"
uv run autorag query --run-id "${RUN_ID}" --mode hybrid --question "${QUESTION}" --top-k "${TOP_K}"
uv run autorag answer --run-id "${RUN_ID}" --mode hybrid --question "${QUESTION}" --top-k "${TOP_K}"
