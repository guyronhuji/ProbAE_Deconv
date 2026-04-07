#!/usr/bin/env bash
# ============================================================
# Launch a suite run on a RunPod pod over SSH (from local).
#
# Usage:
#   bash runpod/run_remote.sh \
#     --ssh-target <user@host> \
#     --identity ~/.ssh/id_ed25519 \
#     --config configs/experiment_suite.yaml \
#     --send
#
# Notes:
# - Expects repo already present on pod at /workspace/ProbAE_Deconv
# - Passes arguments through to runpod/run_suite.sh
# ============================================================

set -euo pipefail

SSH_TARGET=""
IDENTITY_FILE=""
REMOTE_REPO_DIR="/workspace/ProbAE_Deconv"
CONFIG_PATH="configs/experiment_suite.yaml"
TAG=""
SEND_FLAG="--no-send"
DOWNSAMPLE_FACTOR=""
OUTPUT_DIR=""
NOTEBOOK_DIR=""
DATASET_PATH_OVERRIDE=""
BREAST_DATA_DIR=""

usage() {
  cat <<'EOF'
Usage:
  bash runpod/run_remote.sh [options]

Options:
  --ssh-target <user@host>   Required RunPod SSH target
  --identity <path>          Optional SSH private key
  --remote-repo <path>       Remote repo path (default: /workspace/ProbAE_Deconv)
  --config <path>            Config path on remote repo
  --tag <name>               Run tag
  --downsample-factor <int>  Optional quick-test downsample
  --output-dir <path>        Remote output_dir override
  --notebook-dir <path>      Remote notebook_output_dir override
  --dataset-path <path>      Override dataset.input_path (remote path)
  --breast-data-dir <path>   Remote path to breast parquet files (triggers auto-prepare)
  --send                     Send tar via runpodctl at end
  --no-send                  Do not send tar (default)
  -h, --help                 Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh-target) SSH_TARGET="$2"; shift 2 ;;
    --identity) IDENTITY_FILE="$2"; shift 2 ;;
    --remote-repo) REMOTE_REPO_DIR="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --downsample-factor) DOWNSAMPLE_FACTOR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --notebook-dir) NOTEBOOK_DIR="$2"; shift 2 ;;
    --dataset-path) DATASET_PATH_OVERRIDE="$2"; shift 2 ;;
    --breast-data-dir) BREAST_DATA_DIR="$2"; shift 2 ;;
    --send) SEND_FLAG="--send"; shift ;;
    --no-send) SEND_FLAG="--no-send"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [ -z "${SSH_TARGET}" ]; then
  echo "ERROR: --ssh-target is required."
  usage
  exit 1
fi

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)
if [ -n "${IDENTITY_FILE}" ]; then
  SSH_OPTS+=(-i "${IDENTITY_FILE}")
fi

REMOTE_CMD=("bash" "runpod/run_suite.sh" "--config" "${CONFIG_PATH}" "${SEND_FLAG}")
if [ -n "${TAG}" ]; then
  REMOTE_CMD+=("--tag" "${TAG}")
fi
if [ -n "${DOWNSAMPLE_FACTOR}" ]; then
  REMOTE_CMD+=("--downsample-factor" "${DOWNSAMPLE_FACTOR}")
fi
if [ -n "${OUTPUT_DIR}" ]; then
  REMOTE_CMD+=("--output-dir" "${OUTPUT_DIR}")
fi
if [ -n "${NOTEBOOK_DIR}" ]; then
  REMOTE_CMD+=("--notebook-dir" "${NOTEBOOK_DIR}")
fi
if [ -n "${DATASET_PATH_OVERRIDE}" ]; then
  REMOTE_CMD+=("--dataset-path" "${DATASET_PATH_OVERRIDE}")
fi
if [ -n "${BREAST_DATA_DIR}" ]; then
  REMOTE_CMD+=("--breast-data-dir" "${BREAST_DATA_DIR}")
fi

printf -v REMOTE_JOINED ' %q' "${REMOTE_CMD[@]}"
REMOTE_JOINED="${REMOTE_JOINED:1}"

echo "Checking SSH connectivity ..."
ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "echo connected" >/dev/null

echo "Launching remote suite ..."
echo "  target: ${SSH_TARGET}"
echo "  repo:   ${REMOTE_REPO_DIR}"
echo "  cmd:    ${REMOTE_JOINED}"
echo ""

ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "cd ${REMOTE_REPO_DIR} && ${REMOTE_JOINED}"

