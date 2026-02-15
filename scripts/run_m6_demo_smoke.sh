#!/usr/bin/env bash
set -euo pipefail

export AUTORAG_DEMO_RUN_ID="${AUTORAG_DEMO_RUN_ID:-m6_smoke}"
export AUTORAG_DEMO_MODE="${AUTORAG_DEMO_MODE:-vector}"
export AUTORAG_DEMO_TOP_K="${AUTORAG_DEMO_TOP_K:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_m6_demo.sh"
