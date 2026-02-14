#!/usr/bin/env bash
set -euo pipefail

# Fast-path defaults for M6 smoke while preserving external overrides.
export AUTORAG_DEMO_RUN_ID="${AUTORAG_DEMO_RUN_ID:-m6_smoke}"
export AUTORAG_DEMO_MODE="${AUTORAG_DEMO_MODE:-vector}"
export AUTORAG_DEMO_TOP_K="${AUTORAG_DEMO_TOP_K:-4}"
export AUTORAG_EMBEDDING_PROVIDER="${AUTORAG_EMBEDDING_PROVIDER:-local_hash}"
export AUTORAG_EMBEDDING_MODEL="${AUTORAG_EMBEDDING_MODEL:-bge-small-en-v1.5}"
export AUTORAG_RERANKER_ENABLED="${AUTORAG_RERANKER_ENABLED:-false}"
export AUTORAG_RERANKER_MODEL="${AUTORAG_RERANKER_MODEL:-llama3:8b}"
export AUTORAG_RERANKER_CANDIDATE_K="${AUTORAG_RERANKER_CANDIDATE_K:-30}"

bash scripts/run_m6_demo.sh
