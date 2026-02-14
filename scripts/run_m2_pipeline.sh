#!/usr/bin/env bash
set -euo pipefail

uv run autorag ingest --input data/raw/pdfs --run-id m2 --chunking heading_recursive
uv run autorag index-vector --run-id m2 --embedding bge-small-en-v1.5
uv run autorag query --run-id m2 --mode vector --question "How is scope baseline used?" --top-k 8
