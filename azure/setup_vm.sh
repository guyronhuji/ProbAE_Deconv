#!/usr/bin/env bash
# ============================================================
# One-time setup on an Azure GPU VM for ProbAE_Deconv.
#
# Usage (inside VM):
#   bash azure/setup_vm.sh
#
# Optional env vars:
#   REPO_DIR   (default: /workspace/ProbAE_Deconv)
#   REPO_URL   (default: https://github.com/guyronhuji/ProbAE_Deconv.git)
#   REPO_REF   (branch/tag/commit to checkout; optional)
#   VENV_DIR   (default: <REPO_DIR>/.venv)
#   SETUP_LOG_PATH (optional; append setup logs to this file)
# ============================================================

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/ProbAE_Deconv}"
REPO_URL="${REPO_URL:-https://github.com/guyronhuji/ProbAE_Deconv.git}"
REPO_REF="${REPO_REF:-}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
SETUP_LOG_PATH="${SETUP_LOG_PATH:-}"

if [ -n "${SETUP_LOG_PATH}" ]; then
  mkdir -p "$(dirname "${SETUP_LOG_PATH}")"
  touch "${SETUP_LOG_PATH}"
  exec > >(tee -a "${SETUP_LOG_PATH}") 2>&1
  echo "Logging setup output to: ${SETUP_LOG_PATH}"
  echo "Started at: $(date -Iseconds)"
  echo ""
fi

as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "ERROR: command requires root privileges, but sudo is not available: $*"
    exit 1
  fi
}

wait_for_apt_processes() {
  local waited=0
  while pgrep -x apt >/dev/null 2>&1 || pgrep -x apt-get >/dev/null 2>&1 || pgrep -x dpkg >/dev/null 2>&1; do
    echo "Waiting for apt/dpkg lock holders to finish (${waited}s elapsed) ..."
    sleep 5
    waited=$((waited + 5))
  done
}

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
  wait_for_apt_processes
  as_root apt-get update
  DEBIAN_FRONTEND=noninteractive as_root apt-get install -y git python3 python3-pip python3-venv tmux
else
  echo "apt-get not found; skipping tmux installation"
fi

echo ""
echo "=== [3/5] Install package + dependencies ==="
if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -e ".[extras]"
"${VENV_DIR}/bin/python" -m pip install PyCytoData

echo ""
echo "=== [4/5] Verify runtime ==="
"${VENV_DIR}/bin/python" - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
else:
    print("NOTE: CUDA is not available yet. Install GPU drivers or use a CUDA-ready image.")
PY

echo ""
echo "=== [5/5] Done ==="
echo "Next:"
echo "  cd ${REPO_DIR}"
echo "  source ${VENV_DIR}/bin/activate"
echo "  bash runpod/prepare_dataset.sh"
echo "  bash runpod/run_suite.sh --config configs/experiment_suite.yaml --no-send"
