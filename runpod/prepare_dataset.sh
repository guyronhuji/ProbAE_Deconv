#!/usr/bin/env bash
# ============================================================
# Download and prepare Levine32 inside a RunPod pod.
#
# Usage:
#   bash runpod/prepare_dataset.sh
#   bash runpod/prepare_dataset.sh --output data/levine32_processed.h5ad --force-download
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_PATH="data/levine32_processed.h5ad"
LOG_PATH="data/levine32_preprocessing_log.json"
FORCE_DOWNLOAD=0
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  bash runpod/prepare_dataset.sh [options]

Options:
  --output <path>         Output h5ad path (default: data/levine32_processed.h5ad)
  --log-path <path>       Output log JSON path (default: data/levine32_preprocessing_log.json)
  --force-download        Force redownload from source
  --overwrite             Overwrite existing output
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    --log-path) LOG_PATH="$2"; shift 2 ;;
    --force-download) FORCE_DOWNLOAD=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

CMD=(python3 scripts/prepare_levine32_dataset.py --output "$OUTPUT_PATH" --log-path "$LOG_PATH")
if [ "$FORCE_DOWNLOAD" -eq 1 ]; then
  CMD+=(--force-download)
fi
if [ "$OVERWRITE" -eq 1 ]; then
  CMD+=(--overwrite)
fi

echo "Preparing Levine32 dataset ..."
echo "  repo:   ${REPO_ROOT}"
echo "  output: ${OUTPUT_PATH}"
echo "  log:    ${LOG_PATH}"

"${CMD[@]}"

echo ""
echo "Dataset preparation complete."

