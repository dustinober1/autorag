#!/usr/bin/env bash
set -euo pipefail

uv run autorag ingest --input data/raw/pdfs --run-id m3 --chunking heading_recursive
uv run autorag build-kg --run-id m3
uv run autorag query --run-id m3 --mode graph --question "How does scope control affect risk response?" --top-k 8
