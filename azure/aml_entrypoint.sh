#!/usr/bin/env bash
# ============================================================
# Azure ML job entrypoint for ProbAE_Deconv.
# Runs setup + full suite and optionally copies artifacts.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

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
ARTIFACT_DIR=""
SKIP_INSTALL=0
EDITABLE_SPEC=".[extras]"

usage() {
  cat <<'EOF'
Usage:
  bash azure/aml_entrypoint.sh [options]

Options:
  --config <path>            Config YAML (default: configs/experiment_suite.yaml)
  --tag <name>               Run tag (default: azure_<timestamp>)
  --output-dir <path>        Override output_dir in config
  --notebook-dir <path>      Override notebook_output_dir in config
  --downsample-factor <int>  Optional quick-test downsampling
  --dataset-path <path>      Override dataset.input_path in suite config
  --auto-prepare-dataset     Auto-download/prepare Levine32 if missing (default)
  --no-auto-prepare-dataset  Disable automatic dataset preparation
  --dataset-force-download   Force redownload during auto-prepare
  --gpu-parallel <auto|N>    GPU multiprocessing workers (default: auto)
  --gpu-mem-per-job-gb <N>   VRAM budget per parallel GPU job for auto mode (default: 12)
  --artifact-dir <path>      Copy run outputs/notebooks into this folder
  --skip-install             Skip pip installation phase
  --editable-spec <spec>     pip editable install spec (default: .[extras])
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --artifact-dir) ARTIFACT_DIR="$2"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --editable-spec) EDITABLE_SPEC="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "ERROR: config does not exist: ${CONFIG_PATH}"
  exit 1
fi

if [ -z "${TAG}" ]; then
  TAG="azure_$(date +%Y%m%d_%H%M%S)"
fi
if [ -z "${OUTPUT_DIR}" ]; then
  OUTPUT_DIR="outputs/${TAG}"
fi
if [ -z "${NOTEBOOK_DIR}" ]; then
  NOTEBOOK_DIR="notebooks/experiment_suite_${TAG}"
fi

if [ "${SKIP_INSTALL}" -eq 0 ]; then
  echo "Installing dependencies ..."
  python3 -m pip install --upgrade pip
  python3 -m pip install -e "${EDITABLE_SPEC}"
  python3 -m pip install PyCytoData
fi

RUN_CMD=(bash runpod/run_suite.sh --config "${CONFIG_PATH}" --tag "${TAG}" --output-dir "${OUTPUT_DIR}" --notebook-dir "${NOTEBOOK_DIR}" --gpu-parallel "${GPU_PARALLEL}" --gpu-mem-per-job-gb "${GPU_MEM_PER_JOB_GB}" --no-send)
if [ -n "${DOWNSAMPLE_FACTOR}" ]; then
  RUN_CMD+=(--downsample-factor "${DOWNSAMPLE_FACTOR}")
fi
if [ -n "${DATASET_PATH_OVERRIDE}" ]; then
  RUN_CMD+=(--dataset-path "${DATASET_PATH_OVERRIDE}")
fi
if [ "${AUTO_PREPARE_DATASET}" -eq 1 ]; then
  RUN_CMD+=(--auto-prepare-dataset)
else
  RUN_CMD+=(--no-auto-prepare-dataset)
fi
if [ "${DATASET_FORCE_DOWNLOAD}" -eq 1 ]; then
  RUN_CMD+=(--dataset-force-download)
fi

printf -v RUN_JOINED ' %q' "${RUN_CMD[@]}"
RUN_JOINED="${RUN_JOINED:1}"
echo "Running suite command:"
echo "  ${RUN_JOINED}"
echo ""

"${RUN_CMD[@]}"

if [ -n "${ARTIFACT_DIR}" ]; then
  mkdir -p "${ARTIFACT_DIR}"
  if [ -d "${OUTPUT_DIR}" ]; then
    cp -R "${OUTPUT_DIR}" "${ARTIFACT_DIR}/"
  fi
  if [ -d "${NOTEBOOK_DIR}" ]; then
    cp -R "${NOTEBOOK_DIR}" "${ARTIFACT_DIR}/"
  fi
  if [ -f "data/levine32_preprocessing_log.json" ]; then
    cp -f "data/levine32_preprocessing_log.json" "${ARTIFACT_DIR}/"
  fi
  echo "Copied artifacts to: ${ARTIFACT_DIR}"
fi

echo "Azure entrypoint complete."
