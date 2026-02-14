#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${AUTORAG_M4_RUN_ID:-m4}"
INPUT_DIR="${AUTORAG_M4_INPUT_DIR:-data/raw/pdfs}"
QUESTION="${AUTORAG_M4_QUESTION:-Compare mitigation and acceptance strategies.}"
TOP_K="${AUTORAG_M4_TOP_K:-10}"
CHUNKING="${AUTORAG_M4_CHUNKING:-heading_recursive}"
EMBEDDING="${AUTORAG_M4_EMBEDDING:-bge-small-en-v1.5}"

uv run autorag ingest --input "${INPUT_DIR}" --run-id "${RUN_ID}" --chunking "${CHUNKING}"
uv run autorag index-vector --run-id "${RUN_ID}" --embedding "${EMBEDDING}"
uv run autorag build-kg --run-id "${RUN_ID}"
uv run autorag query --run-id "${RUN_ID}" --mode hybrid --question "${QUESTION}" --top-k "${TOP_K}"
uv run autorag answer --run-id "${RUN_ID}" --question "${QUESTION}" --mode hybrid --top-k "${TOP_K}"
