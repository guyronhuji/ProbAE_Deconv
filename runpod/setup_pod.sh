#!/usr/bin/env bash
# ============================================================
# One-time setup inside a RunPod pod for ProbAE_Deconv.
#
# Usage (inside pod):
#   export REPO_URL="https://github.com/<you>/<repo>.git"   # required on first run
#   bash runpod/setup_pod.sh
#
# Optional env vars:
#   REPO_DIR   (default: /workspace/ProbAE_Deconv)
#   REPO_URL   (required if REPO_DIR does not exist yet)
#   REPO_REF   (branch/tag/commit to checkout; optional)
# ============================================================

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/ProbAE_Deconv}"
REPO_URL="${REPO_URL:-}"
REPO_REF="${REPO_REF:-}"

echo "=== [1/4] Prepare repo ==="
if [ ! -d "${REPO_DIR}/.git" ]; then
  if [ -z "${REPO_URL}" ]; then
    echo "ERROR: REPO_URL is required for first-time setup."
    echo "Example:"
    echo "  REPO_URL=https://github.com/<you>/<repo>.git bash runpod/setup_pod.sh"
    exit 1
  fi
  git clone "${REPO_URL}" "${REPO_DIR}"
else
  echo "Repo already exists at ${REPO_DIR}"
fi

cd "${REPO_DIR}"
if [ -n "${REPO_REF}" ]; then
  git fetch --all --tags
  git checkout "${REPO_REF}"
fi

echo ""
echo "=== [2/4] Install package + dependencies ==="
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[extras]"

echo ""
echo "=== [3/4] Verify runtime ==="
python3 - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
PY

echo ""
echo "=== [4/4] Done ==="
echo "Next:"
echo "  cd ${REPO_DIR}"
echo "  bash runpod/run_suite.sh --config configs/experiment_suite.yaml --send"

