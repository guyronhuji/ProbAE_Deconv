#!/usr/bin/env bash
# ============================================================
# Prepare breast CyTOF dataset inside a RunPod pod.
# Merges normalized_not_scaled parquet files → h5ad.
#
# Usage (inside pod):
#   bash runpod/prepare_breast_dataset.sh --input-dir /path/to/parquets
#
# The parquet files must already be present on the pod.
# Transfer from local machine:
#   Local:  runpodctl send /path/to/for_guy
#   Pod:    runpodctl receive <TRANSFER_CODE>
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_DIR=""
OUTPUT_PATH="data/breast_cytof_processed.h5ad"
FILE_VARIANT="normalized_not_scaled"
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  bash runpod/prepare_breast_dataset.sh [options]

Options:
  --input-dir <path>      Required: directory with normalized_not_scaled_*.parquet files
  --output <path>         Output h5ad path (default: data/breast_cytof_processed.h5ad)
  --file-variant <name>   Parquet filename prefix (default: normalized_not_scaled)
  --overwrite             Overwrite existing output
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    --file-variant) FILE_VARIANT="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [ -z "${INPUT_DIR}" ]; then
  echo "ERROR: --input-dir is required."
  usage
  exit 1
fi

CMD=(python3 scripts/prepare_breast_dataset.py
  --input-dir "${INPUT_DIR}"
  --output "${OUTPUT_PATH}"
  --file-variant "${FILE_VARIANT}")
if [ "${OVERWRITE}" -eq 1 ]; then
  CMD+=(--overwrite)
fi

echo "Preparing breast CyTOF dataset ..."
echo "  repo:        ${REPO_ROOT}"
echo "  input dir:   ${INPUT_DIR}"
echo "  output:      ${OUTPUT_PATH}"

"${CMD[@]}"

echo ""
echo "Dataset preparation complete."
