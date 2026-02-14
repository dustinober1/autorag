#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${AUTORAG_BOOTSTRAP_SOURCE_DIR:-data/fixtures/pdfs}"
OUTPUT_DIR="${AUTORAG_BOOTSTRAP_OUTPUT_DIR:-data/raw/pdfs}"

FILES=(
  "project_scope_fixture.pdf"
  "risk_response_fixture.pdf"
)

expected_sha256() {
  local filename="$1"
  case "${filename}" in
    "project_scope_fixture.pdf")
      echo "589ef4772fa4e9b403466fea66347cbc0a0cc7f610e15bf141236f361c471eed"
      ;;
    "risk_response_fixture.pdf")
      echo "0c774ba93c9b5727f7c4ae44f121d11bde8563081894cfaee1dddd36c1b29efd"
      ;;
    *)
      echo "Unknown fixture: ${filename}" >&2
      return 1
      ;;
  esac
}

mkdir -p "${OUTPUT_DIR}"

for filename in "${FILES[@]}"; do
  source_path="${SOURCE_DIR}/${filename}"
  destination_path="${OUTPUT_DIR}/${filename}"
  if [[ ! -f "${source_path}" ]]; then
    echo "Missing fixture file: ${source_path}" >&2
    exit 1
  fi

  actual_sha256="$(shasum -a 256 "${source_path}" | awk '{print $1}')"
  expected="$(expected_sha256 "${filename}")"
  if [[ "${actual_sha256}" != "${expected}" ]]; then
    echo "Checksum mismatch for ${source_path}" >&2
    echo "Expected: ${expected}" >&2
    echo "Actual:   ${actual_sha256}" >&2
    exit 1
  fi

  cp "${source_path}" "${destination_path}"
done

echo "Bootstrapped ${#FILES[@]} fixture PDFs into ${OUTPUT_DIR}"
