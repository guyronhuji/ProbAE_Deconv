#!/usr/bin/env bash
# ============================================================
# Submit ProbAE full suite as an Azure ML command job.
#
# Prerequisites:
#   - az cli installed
#   - az extension add -n ml
#   - az login
#   - existing Azure ML workspace + compute target
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SUBSCRIPTION_ID=""
RESOURCE_GROUP=""
WORKSPACE_NAME=""
COMPUTE_NAME=""
EXPERIMENT_NAME="probae-deconv"
JOB_NAME=""
CONFIG_PATH="configs/experiment_suite.yaml"
TAG=""
OUTPUT_DIR=""
NOTEBOOK_DIR=""
DOWNSAMPLE_FACTOR=""
DATASET_PATH_OVERRIDE=""
AUTO_PREPARE_DATASET=1
DATASET_FORCE_DOWNLOAD=0
GPU_PARALLEL="auto"
GPU_MEM_PER_JOB_GB="12"
AML_IMAGE="mcr.microsoft.com/azureml/openmpi4.1.0-cuda12.2-cudnn9-ubuntu22.04"
STREAM_LOGS=1

usage() {
  cat <<'EOF'
Usage:
  bash azure/submit_aml_job.sh [options]

Required:
  --resource-group <name>    Azure resource group
  --workspace <name>         Azure ML workspace name
  --compute <name>           Azure ML compute target (existing)

Optional:
  --subscription <id>        Azure subscription id/name
  --experiment <name>        AML experiment name (default: probae-deconv)
  --job-name <name>          AML job name (default: probae-<timestamp>)
  --config <path>            Config YAML in repo (default: configs/experiment_suite.yaml)
  --tag <name>               Run tag (default: <job-name>)
  --output-dir <path>        Override output_dir in config
  --notebook-dir <path>      Override notebook_output_dir in config
  --downsample-factor <int>  Optional quick-test downsampling
  --dataset-path <path>      Override dataset.input_path in suite config
  --auto-prepare-dataset     Auto-download/prepare Levine32 if missing (default)
  --no-auto-prepare-dataset  Disable automatic dataset preparation
  --dataset-force-download   Force redownload during auto-prepare
  --gpu-parallel <auto|N>    GPU multiprocessing workers (default: auto)
  --gpu-mem-per-job-gb <N>   VRAM budget per parallel GPU job for auto mode (default: 12)
  --image <uri>              AML runtime image URI
  --stream                   Stream logs after submission (default)
  --no-stream                Do not stream logs
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subscription) SUBSCRIPTION_ID="$2"; shift 2 ;;
    --resource-group) RESOURCE_GROUP="$2"; shift 2 ;;
    --workspace) WORKSPACE_NAME="$2"; shift 2 ;;
    --compute) COMPUTE_NAME="$2"; shift 2 ;;
    --experiment) EXPERIMENT_NAME="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --notebook-dir) NOTEBOOK_DIR="$2"; shift 2 ;;
    --downsample-factor) DOWNSAMPLE_FACTOR="$2"; shift 2 ;;
    --dataset-path) DATASET_PATH_OVERRIDE="$2"; shift 2 ;;
    --auto-prepare-dataset) AUTO_PREPARE_DATASET=1; shift ;;
    --no-auto-prepare-dataset) AUTO_PREPARE_DATASET=0; shift ;;
    --dataset-force-download) DATASET_FORCE_DOWNLOAD=1; shift ;;
    --gpu-parallel) GPU_PARALLEL="$2"; shift 2 ;;
    --gpu-mem-per-job-gb) GPU_MEM_PER_JOB_GB="$2"; shift 2 ;;
    --image) AML_IMAGE="$2"; shift 2 ;;
    --stream) STREAM_LOGS=1; shift ;;
    --no-stream) STREAM_LOGS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [ -z "${RESOURCE_GROUP}" ] || [ -z "${WORKSPACE_NAME}" ] || [ -z "${COMPUTE_NAME}" ]; then
  echo "ERROR: --resource-group, --workspace, and --compute are required."
  usage
  exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "ERROR: config does not exist: ${CONFIG_PATH}"
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

if ! az ml workspace show --resource-group "${RESOURCE_GROUP}" --name "${WORKSPACE_NAME}" >/dev/null 2>&1; then
  echo "ERROR: Azure ML workspace not found: ${WORKSPACE_NAME} (resource group: ${RESOURCE_GROUP})"
  exit 1
fi

if ! az ml compute show --resource-group "${RESOURCE_GROUP}" --workspace-name "${WORKSPACE_NAME}" --name "${COMPUTE_NAME}" >/dev/null 2>&1; then
  echo "ERROR: Azure ML compute target not found: ${COMPUTE_NAME}"
  exit 1
fi

if [ -z "${JOB_NAME}" ]; then
  JOB_NAME="probae-$(date +%Y%m%d-%H%M%S)"
fi
if [ -z "${TAG}" ]; then
  TAG="${JOB_NAME}"
fi
if [ -z "${OUTPUT_DIR}" ]; then
  OUTPUT_DIR="outputs/${TAG}"
fi
if [ -z "${NOTEBOOK_DIR}" ]; then
  NOTEBOOK_DIR="notebooks/experiment_suite_${TAG}"
fi

ENTRY_ARGS=(--config "${CONFIG_PATH}" --tag "${TAG}" --output-dir "${OUTPUT_DIR}" --notebook-dir "${NOTEBOOK_DIR}" --gpu-parallel "${GPU_PARALLEL}" --gpu-mem-per-job-gb "${GPU_MEM_PER_JOB_GB}" --artifact-dir '${{outputs.suite_artifacts}}')
if [ -n "${DOWNSAMPLE_FACTOR}" ]; then
  ENTRY_ARGS+=(--downsample-factor "${DOWNSAMPLE_FACTOR}")
fi
if [ -n "${DATASET_PATH_OVERRIDE}" ]; then
  ENTRY_ARGS+=(--dataset-path "${DATASET_PATH_OVERRIDE}")
fi
if [ "${AUTO_PREPARE_DATASET}" -eq 1 ]; then
  ENTRY_ARGS+=(--auto-prepare-dataset)
else
  ENTRY_ARGS+=(--no-auto-prepare-dataset)
fi
if [ "${DATASET_FORCE_DOWNLOAD}" -eq 1 ]; then
  ENTRY_ARGS+=(--dataset-force-download)
fi

printf -v ENTRY_JOINED ' %q' "${ENTRY_ARGS[@]}"
ENTRY_JOINED="${ENTRY_JOINED:1}"

TMP_JOB_YAML="/tmp/probae_aml_${JOB_NAME}.yaml"
cat > "${TMP_JOB_YAML}" <<EOF
\$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
type: command
name: ${JOB_NAME}
display_name: ${JOB_NAME}
experiment_name: ${EXPERIMENT_NAME}
code: .
command: >-
  bash azure/aml_entrypoint.sh ${ENTRY_JOINED}
environment:
  image: ${AML_IMAGE}
compute: azureml:${COMPUTE_NAME}
outputs:
  suite_artifacts:
    type: uri_folder
EOF

echo "Submitting Azure ML job ..."
echo "  resource_group: ${RESOURCE_GROUP}"
echo "  workspace:      ${WORKSPACE_NAME}"
echo "  compute:        ${COMPUTE_NAME}"
echo "  experiment:     ${EXPERIMENT_NAME}"
echo "  job_name:       ${JOB_NAME}"
echo "  temp_yaml:      ${TMP_JOB_YAML}"
echo ""

JOB_JSON="$(
  az ml job create \
    --resource-group "${RESOURCE_GROUP}" \
    --workspace-name "${WORKSPACE_NAME}" \
    --file "${TMP_JOB_YAML}" \
    -o json
)"

SUBMITTED_JOB_NAME="$(
python3 - <<PY
import json
raw = """${JOB_JSON}"""
d = json.loads(raw)
print(d.get("name", ""))
PY
)"

STUDIO_URL="$(
python3 - <<PY
import json
raw = """${JOB_JSON}"""
d = json.loads(raw)
print(d.get("studio_url", ""))
PY
)"

echo "Submitted job: ${SUBMITTED_JOB_NAME}"
if [ -n "${STUDIO_URL}" ]; then
  echo "Studio URL: ${STUDIO_URL}"
fi

if [ "${STREAM_LOGS}" -eq 1 ]; then
  echo ""
  echo "Streaming logs ..."
  az ml job stream \
    --resource-group "${RESOURCE_GROUP}" \
    --workspace-name "${WORKSPACE_NAME}" \
    --name "${SUBMITTED_JOB_NAME}"
fi

echo ""
echo "To fetch outputs:"
echo "  bash azure/fetch_results.sh --resource-group ${RESOURCE_GROUP} --workspace ${WORKSPACE_NAME} --job-name ${SUBMITTED_JOB_NAME}"
