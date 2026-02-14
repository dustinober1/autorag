#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${AUTORAG_BOOTSTRAP_SOURCE_DIR:-data/fixtures/pdfs}"
OUTPUT_DIR="${AUTORAG_BOOTSTRAP_OUTPUT_DIR:-data/raw/pdfs}"
PMBOK_SOURCE_PATH="${AUTORAG_PMBOK_SOURCE_PATH:-files/PMP/pmbokguide_eighthed_eng.pdf}"
INCLUDE_PMBOK="${AUTORAG_INCLUDE_PMBOK:-0}"
INCLUDE_PMBOK_NORMALIZED="$(printf '%s' "${INCLUDE_PMBOK}" | tr '[:upper:]' '[:lower:]')"

FILES=(
  "project_scope_fixture.pdf"
  "risk_response_fixture.pdf"
)

case "${INCLUDE_PMBOK_NORMALIZED}" in
  1|true|yes|on)
    FILES+=("pmbokguide_eighthed_eng.pdf")
    ;;
esac

expected_sha256() {
  local filename="$1"
  case "${filename}" in
    "project_scope_fixture.pdf")
      echo "589ef4772fa4e9b403466fea66347cbc0a0cc7f610e15bf141236f361c471eed"
      ;;
    "risk_response_fixture.pdf")
      echo "0c774ba93c9b5727f7c4ae44f121d11bde8563081894cfaee1dddd36c1b29efd"
      ;;
    "pmbokguide_eighthed_eng.pdf")
      echo "f0bc76467317c58f2d520ac98a59b5b415c62a7318bdc1ba7be6e75c4b208135"
      ;;
    *)
      echo "Unknown fixture: ${filename}" >&2
      return 1
      ;;
  esac
}

source_path_for_file() {
  local filename="$1"
  case "${filename}" in
    "pmbokguide_eighthed_eng.pdf")
      echo "${PMBOK_SOURCE_PATH}"
      ;;
    *)
      echo "${SOURCE_DIR}/${filename}"
      ;;
  esac
}

mkdir -p "${OUTPUT_DIR}"

for filename in "${FILES[@]}"; do
  source_path="$(source_path_for_file "${filename}")"
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
