#!/usr/bin/env bash
set -euo pipefail

# Fast-path defaults for M6 smoke while preserving external overrides.
export AUTORAG_DEMO_RUN_ID="${AUTORAG_DEMO_RUN_ID:-m6_smoke}"
export AUTORAG_DEMO_MODE="${AUTORAG_DEMO_MODE:-vector}"
export AUTORAG_DEMO_TOP_K="${AUTORAG_DEMO_TOP_K:-4}"

bash scripts/run_m6_demo.sh
