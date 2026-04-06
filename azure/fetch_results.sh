#!/usr/bin/env bash
# ============================================================
# Download Azure ML job outputs for ProbAE_Deconv.
# ============================================================

set -euo pipefail

SUBSCRIPTION_ID=""
RESOURCE_GROUP=""
WORKSPACE_NAME=""
JOB_NAME=""
DEST_ROOT="./outputs/azure_jobs"

usage() {
  cat <<'EOF'
Usage:
  bash azure/fetch_results.sh [options]

Required:
  --resource-group <name>    Azure resource group
  --workspace <name>         Azure ML workspace name
  --job-name <name>          Azure ML job name

Optional:
  --subscription <id>        Azure subscription id/name
  --dest <path>              Local destination root (default: ./outputs/azure_jobs)
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subscription) SUBSCRIPTION_ID="$2"; shift 2 ;;
    --resource-group) RESOURCE_GROUP="$2"; shift 2 ;;
    --workspace) WORKSPACE_NAME="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --dest) DEST_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [ -z "${RESOURCE_GROUP}" ] || [ -z "${WORKSPACE_NAME}" ] || [ -z "${JOB_NAME}" ]; then
  echo "ERROR: --resource-group, --workspace, and --job-name are required."
  usage
  exit 1
fi

if ! command -v az >/dev/null 2>&1; then
  echo "ERROR: az cli is not installed."
  exit 1
fi

if ! az extension show -n ml >/dev/null 2>&1; then
  echo "Installing Azure ML extension ..."
  az extension add -n ml -y >/dev/null
fi

if [ -n "${SUBSCRIPTION_ID}" ]; then
  az account set --subscription "${SUBSCRIPTION_ID}"
fi

TARGET_DIR="${DEST_ROOT}/${JOB_NAME}"
mkdir -p "${TARGET_DIR}"

echo "Downloading outputs ..."
echo "  resource_group: ${RESOURCE_GROUP}"
echo "  workspace:      ${WORKSPACE_NAME}"
echo "  job_name:       ${JOB_NAME}"
echo "  destination:    ${TARGET_DIR}"
echo ""

if az ml job download \
  --resource-group "${RESOURCE_GROUP}" \
  --workspace-name "${WORKSPACE_NAME}" \
  --name "${JOB_NAME}" \
  --all \
  --download-path "${TARGET_DIR}" >/dev/null 2>&1; then
  :
else
  az ml job download \
    --resource-group "${RESOURCE_GROUP}" \
    --workspace-name "${WORKSPACE_NAME}" \
    --name "${JOB_NAME}" \
    --download-path "${TARGET_DIR}"
fi

echo "Download complete: ${TARGET_DIR}"
