#!/usr/bin/env bash
set -euo pipefail

export AUTORAG_EMBEDDING_PROVIDER="${AUTORAG_EMBEDDING_PROVIDER:-ollama}"
export AUTORAG_EMBEDDING_MODEL="${AUTORAG_EMBEDDING_MODEL:-embeddinggemma:300m}"
export AUTORAG_RERANKER_ENABLED="${AUTORAG_RERANKER_ENABLED:-true}"
export AUTORAG_RERANKER_MODEL="${AUTORAG_RERANKER_MODEL:-llama3:8b}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_m6_demo.sh"
