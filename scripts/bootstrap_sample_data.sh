#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FIXTURE_DIR="${AUTORAG_FIXTURE_DIR:-${REPO_ROOT}/data/fixtures/pdfs}"
RAW_DIR="${AUTORAG_BOOTSTRAP_RAW_DIR:-${REPO_ROOT}/data/raw/pdfs}"

mkdir -p "${RAW_DIR}"

for fixture in project_scope_fixture.pdf risk_response_fixture.pdf; do
  src="${FIXTURE_DIR}/${fixture}"
  dest="${RAW_DIR}/${fixture}"
  if [[ ! -f "${src}" ]]; then
    echo "Missing fixture: ${src}" >&2
    exit 1
  fi
  cp "${src}" "${dest}"
  echo "Bootstrapped ${fixture} -> ${dest}"
done
