#!/usr/bin/env bash
# ============================================================
# Run the full ProbAE experiment suite inside a RunPod pod.
#
# Usage (inside pod):
#   bash runpod/run_suite.sh --config configs/experiment_suite.yaml --send
#
# Key behavior:
# - clones nothing (expects repo already present in pod)
# - writes a temporary RunPod-tuned config
# - forces NN methods to CUDA
# - runs full suite
# - optionally packs outputs and prints/starts runpodctl send
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="configs/experiment_suite.yaml"
TAG=""
SEND_RESULTS=0
DOWNSAMPLE_FACTOR=""
OUTPUT_DIR=""
NOTEBOOK_DIR=""
GPU_PARALLEL="auto"
GPU_MEM_PER_JOB_GB="12"

usage() {
  cat <<'EOF'
Usage:
  bash runpod/run_suite.sh [options]

Options:
  --config <path>            Config YAML (default: configs/experiment_suite.yaml)
  --tag <name>               Run tag (default: runpod_<timestamp>)
  --output-dir <path>        Override output_dir in config
  --notebook-dir <path>      Override notebook_output_dir in config
  --downsample-factor <int>  Optional quick-test downsampling (e.g. 5, 10, 20)
  --gpu-parallel <auto|N>    GPU multiprocessing workers (default: auto)
  --gpu-mem-per-job-gb <N>   VRAM budget per parallel GPU job for auto mode (default: 12)
  --send                     Create tar.gz and run `runpodctl send` at end
  --no-send                  Do not send results (default)
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
    --gpu-parallel) GPU_PARALLEL="$2"; shift 2 ;;
    --gpu-mem-per-job-gb) GPU_MEM_PER_JOB_GB="$2"; shift 2 ;;
    --send) SEND_RESULTS=1; shift ;;
    --no-send) SEND_RESULTS=0; shift ;;
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
  TAG="runpod_$(date +%Y%m%d_%H%M%S)"
fi
if [ -z "${OUTPUT_DIR}" ]; then
  OUTPUT_DIR="outputs/${TAG}"
fi
if [ -z "${NOTEBOOK_DIR}" ]; then
  NOTEBOOK_DIR="notebooks/experiment_suite_${TAG}"
fi

resolve_gpu_workers() {
  local mode="$1"
  local mem_per_job="$2"
  local workers=1
  if [[ "$mode" =~ ^[0-9]+$ ]]; then
    workers="$mode"
  elif [ "$mode" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      local free_mb
      free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
      if [[ "${free_mb}" =~ ^[0-9]+$ ]] && [[ "${mem_per_job}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        workers="$(
          python3 - <<PY
import math
free_mb = float("${free_mb}")
mem_per_job_gb = float("${mem_per_job}")
need_mb = max(1.0, mem_per_job_gb * 1024.0)
w = int(max(1, math.floor(free_mb / need_mb)))
print(min(w, 8))
PY
)"
      fi
    fi
  else
    echo "ERROR: --gpu-parallel must be 'auto' or a positive integer"
    exit 1
  fi

  if ! [[ "$workers" =~ ^[0-9]+$ ]] || [ "$workers" -lt 1 ]; then
    workers=1
  fi
  echo "$workers"
}

GPU_WORKERS="$(resolve_gpu_workers "${GPU_PARALLEL}" "${GPU_MEM_PER_JOB_GB}")"

TMP_CFG="/tmp/probae_runpod_${TAG}.yaml"
LOG_DIR="${OUTPUT_DIR}/reports"
mkdir -p "${LOG_DIR}"

echo "Preparing RunPod config ..."
python3 - <<PY
from pathlib import Path
import yaml

cfg_path = Path("${CONFIG_PATH}")
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

cfg["output_dir"] = "${OUTPUT_DIR}"
cfg["notebook_output_dir"] = "${NOTEBOOK_DIR}"
cfg["show_progress"] = True
cfg["show_run_logs"] = True
cfg["show_training_progress"] = True
cfg["training_progress_level"] = cfg.get("training_progress_level", "epoch")
cfg["gpu_multiprocessing_workers"] = int("${GPU_WORKERS}")
cfg["gpu_parallel_methods"] = ["deterministic_archetypal_ae", "probabilistic_archetypal_ae", "ae", "vae"]
if int("${GPU_WORKERS}") > 1:
    # Multiple concurrent NN jobs make nested progress bars unreadable.
    cfg["show_training_progress"] = False

methods = cfg.setdefault("methods", {})
for name in ("deterministic_archetypal_ae", "probabilistic_archetypal_ae", "ae", "vae"):
    m = methods.setdefault(name, {})
    m["device"] = "cuda"
    methods[name] = m
cfg["methods"] = methods

ds = cfg.setdefault("dataset", {})
raw_ds = "${DOWNSAMPLE_FACTOR}".strip()
if raw_ds:
    ds["downsample_factor"] = int(raw_ds)

Path("${TMP_CFG}").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("Wrote:", "${TMP_CFG}")
print("output_dir:", cfg["output_dir"])
print("notebook_output_dir:", cfg["notebook_output_dir"])
print("downsample_factor:", cfg.get("dataset", {}).get("downsample_factor"))
print("gpu_multiprocessing_workers:", cfg.get("gpu_multiprocessing_workers"))
print("show_training_progress:", cfg.get("show_training_progress"))
PY

echo ""
echo "Running full suite ..."
echo "  repo:   ${REPO_ROOT}"
echo "  config: ${TMP_CFG}"
echo "  log:    ${LOG_DIR}/runpod_stdout.log"
echo ""

python3 scripts/run_experiment_suite.py --config "${TMP_CFG}" 2>&1 | tee "${LOG_DIR}/runpod_stdout.log"

echo ""
echo "Suite completed."
echo "Outputs: ${OUTPUT_DIR}"

if [ "${SEND_RESULTS}" -eq 1 ]; then
  if ! command -v runpodctl >/dev/null 2>&1; then
    echo "runpodctl not found. Skipping send step."
    exit 0
  fi

  TAR_PATH="/tmp/${TAG}_outputs.tar.gz"
  echo "Packing results to ${TAR_PATH} ..."
  tar czf "${TAR_PATH}" -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")"

  echo ""
  echo "Sending results with runpodctl ..."
  echo "Paste the printed code into local runpod/fetch_results.sh"
  runpodctl send "${TAR_PATH}"
fi
