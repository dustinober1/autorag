#!/usr/bin/env bash
set -euo pipefail

uv run autorag smoke --input data/fixtures/pdfs --question "What is project scope?" --run-id m1
