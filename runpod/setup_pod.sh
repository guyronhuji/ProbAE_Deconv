#!/usr/bin/env bash
# ============================================================
# One-time setup inside a RunPod pod for ProbAE_Deconv.
#
# Usage (inside pod):
#   bash runpod/setup_pod.sh
#
# Optional env vars:
#   REPO_DIR   (default: /workspace/ProbAE_Deconv)
#   REPO_URL   (default: https://github.com/guyronhuji/ProbAE_Deconv.git)
#   REPO_REF   (branch/tag/commit to checkout; optional)
# ============================================================

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/ProbAE_Deconv}"
REPO_URL="${REPO_URL:-https://github.com/guyronhuji/ProbAE_Deconv.git}"
REPO_REF="${REPO_REF:-}"

echo "=== [1/5] Prepare repo ==="
if [ ! -d "${REPO_DIR}/.git" ]; then
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
echo "=== [2/5] Install system tools ==="
if command -v apt-get >/dev/null 2>&1; then
  if ! command -v tmux >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y tmux
  else
    echo "tmux already installed"
  fi
else
  echo "apt-get not found; skipping tmux installation"
fi

echo ""
echo "=== [3/5] Install package + dependencies ==="
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[extras]"
python3 -m pip install PyCytoData

echo ""
echo "=== [4/5] Verify runtime ==="
python3 - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
PY

echo ""
echo "=== [5/5] Done ==="
echo "Next:"
echo "  cd ${REPO_DIR}"
echo "  bash runpod/prepare_dataset.sh"
echo "  bash runpod/run_suite.sh --config configs/experiment_suite.yaml --send"
