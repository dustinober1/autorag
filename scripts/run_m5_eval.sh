#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${AUTORAG_M5_RUN_ID:-m5}"
SOURCE_RUN_ID="${AUTORAG_M5_SOURCE_RUN_ID:-m4}"
INPUT_DIR="${AUTORAG_M5_INPUT_DIR:-data/artifacts/${SOURCE_RUN_ID}}"
TARGET_SIZE="${AUTORAG_M5_TARGET_SIZE:-300}"
MATRIX_CONFIG="${AUTORAG_M5_MATRIX_CONFIG:-configs/experiments/matrix.yaml}"
STARTER_DATASET_OUT="${AUTORAG_M5_STARTER_DATASET_OUT:-eval/datasets/starter_questions_20.jsonl}"

uv run autorag eval bootstrap-starter --out "${STARTER_DATASET_OUT}"
uv run autorag eval generate --run-id "${RUN_ID}" --input "${INPUT_DIR}" --target-size "${TARGET_SIZE}"
uv run autorag eval run-matrix --run-id "${RUN_ID}" --config "${MATRIX_CONFIG}"
uv run autorag eval report --run-id "${RUN_ID}"
