#!/usr/bin/env bash
# ============================================================
# Fetch RunPod results archive sent via `runpodctl send`.
#
# Typical flow:
# 1) In pod:
#      bash runpod/run_suite.sh --config configs/experiment_suite.yaml --send
# 2) On local machine:
#      bash runpod/fetch_results.sh
#
# The script receives a .tar.gz and extracts it under outputs/runpod/.
# ============================================================

set -euo pipefail

DEST_ROOT="./outputs/runpod"
TRANSFER_CODE=""

usage() {
  cat <<'EOF'
Usage:
  bash runpod/fetch_results.sh [options]

Options:
  --code <transfer-code>   Transfer code from `runpodctl send`
  --dest <local-dir>       Local destination root (default: ./outputs/runpod)
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --code) TRANSFER_CODE="$2"; shift 2 ;;
    --dest) DEST_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if ! command -v runpodctl >/dev/null 2>&1; then
  echo "ERROR: runpodctl is required. Install with:"
  echo "  brew install runpod/runpodctl/runpodctl"
  exit 1
fi

if [ -z "${TRANSFER_CODE}" ]; then
  echo "Fetching running pods ..."
  PODS_JSON="$(runpodctl pod list 2>/dev/null || true)"
  if [ -n "${PODS_JSON}" ] && [ "${PODS_JSON}" != "null" ]; then
    python3 - <<'PY'
import json, subprocess

raw = subprocess.check_output(["runpodctl", "pod", "list"], text=True)
pods = json.loads(raw)
running = [p for p in pods if (p.get("desiredStatus", "") or "").upper() == "RUNNING"]
if not running:
    print("No running pods found.")
else:
    print("")
    print(f"{'#':<4} {'Pod ID':<20} {'Name':<16} {'GPU'}")
    print(f"{'---':<4} {'-'*20:<20} {'-'*16:<16} {'-'*30}")
    for i, p in enumerate(running, start=1):
        pod_id = p.get("id", "?")
        name = p.get("name", "?")
        machine = p.get("machine") or {}
        gpu = machine.get("gpuDisplayName", "?")
        print(f"{i:<4} {pod_id:<20} {name:<16} {gpu}")
PY
  fi

  echo ""
  echo "In your pod, send results with one of these:"
  echo "  bash runpod/run_suite.sh --config configs/experiment_suite.yaml --send"
  echo "or, manually:"
  echo "  tar czf /tmp/probae_outputs.tar.gz -C /workspace/ProbAE_Deconv outputs/experiment_suite"
  echo "  runpodctl send /tmp/probae_outputs.tar.gz"
  echo ""
  read -rp "Paste transfer code: " TRANSFER_CODE
fi

if [ -z "${TRANSFER_CODE}" ]; then
  echo "No transfer code provided."
  exit 1
fi

mkdir -p "${DEST_ROOT}"
RECV_DIR="${DEST_ROOT}/_incoming_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RECV_DIR}"

echo "Receiving into ${RECV_DIR} ..."
(
  cd "${RECV_DIR}"
  runpodctl receive "${TRANSFER_CODE}"
)

# Find received archive.
TAR_FILE="$(find "${RECV_DIR}" -maxdepth 1 -type f \( -name '*.tar.gz' -o -name '*.tgz' \) | head -1 || true)"
if [ -z "${TAR_FILE}" ]; then
  echo "No tar archive found in ${RECV_DIR}."
  echo "Received files:"
  ls -la "${RECV_DIR}"
  exit 1
fi

BASE_NAME="$(basename "${TAR_FILE}" .tar.gz)"
BASE_NAME="${BASE_NAME%.tgz}"
FINAL_DIR="${DEST_ROOT}/${BASE_NAME}"
mkdir -p "${FINAL_DIR}"

echo "Extracting ${TAR_FILE} -> ${FINAL_DIR} ..."
tar xzf "${TAR_FILE}" -C "${FINAL_DIR}"

echo ""
echo "Done."
echo "Fetched results at: ${FINAL_DIR}"
echo "If archive includes nested paths, inspect:"
echo "  ls -la ${FINAL_DIR}"

