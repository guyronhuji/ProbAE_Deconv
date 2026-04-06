#!/usr/bin/env bash
# ============================================================
# Deploy a fresh Azure GPU VM and bootstrap ProbAE_Deconv.
#
# This script always creates a new VM instance (timestamp-based
# name by default), installs NVIDIA driver extension, and runs
# azure/setup_vm.sh remotely to prepare the environment.
# ============================================================

set -euo pipefail

SUBSCRIPTION_ID=""
RESOURCE_GROUP=""
LOCATION="eastus"
VM_NAME=""
VM_SIZE="Standard_NC4as_T4_v3"
IMAGE_URN="Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest"
ADMIN_USERNAME="azureuser"
SSH_PUBLIC_KEY="${HOME}/.ssh/id_ed25519.pub"
OS_DISK_SIZE_GB="256"
REPO_URL="https://github.com/guyronhuji/ProbAE_Deconv.git"
REPO_REF=""
TAGS="project=probae-deconv"
AUTO_BOOTSTRAP=1
INSTALL_GPU_DRIVER=1
BOOTSTRAP_LOG_PATH=""
LOCAL_BOOTSTRAP_LOG_DIR="./outputs/azure_bootstrap_logs"
SAVE_LOCAL_BOOTSTRAP_LOG=1

usage() {
  cat <<'EOF'
Usage:
  bash azure/deploy_vm.sh [options]

Required:
  --resource-group <name>       Azure resource group

Optional:
  --subscription <id>           Azure subscription id/name
  --location <region>           Azure region (default: eastus)
  --name <vm-name>              VM name (default: probae-<timestamp>)
  --size <vm-size>              VM SKU (default: Standard_NC4as_T4_v3)
  --image <urn>                 VM image URN
  --admin-user <name>           VM admin username (default: azureuser)
  --ssh-public-key <path|key>   Public key path or key text
  --os-disk-gb <int>            OS disk size in GB (default: 256)
  --repo-url <url>              Repo URL to clone
  --repo-ref <git-ref>          Branch/tag/commit to checkout
  --tags "<k=v ...>"            Azure tags passed to `az vm create`
  --bootstrap-log-path <path>   Log file path on VM for bootstrap output
  --local-bootstrap-log-dir <path> Local directory to save run-command output
  --no-local-bootstrap-log      Do not save local bootstrap log
  --no-bootstrap                Skip remote setup_vm bootstrap
  --no-gpu-driver               Skip NVIDIA driver extension install
  -h, --help                    Show this help

Examples:
  bash azure/deploy_vm.sh --resource-group my-rg

  bash azure/deploy_vm.sh \
    --resource-group my-rg \
    --location westeurope \
    --size Standard_NC8as_T4_v3 \
    --repo-ref main
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subscription) SUBSCRIPTION_ID="$2"; shift 2 ;;
    --resource-group) RESOURCE_GROUP="$2"; shift 2 ;;
    --location) LOCATION="$2"; shift 2 ;;
    --name) VM_NAME="$2"; shift 2 ;;
    --size) VM_SIZE="$2"; shift 2 ;;
    --image) IMAGE_URN="$2"; shift 2 ;;
    --admin-user) ADMIN_USERNAME="$2"; shift 2 ;;
    --ssh-public-key) SSH_PUBLIC_KEY="$2"; shift 2 ;;
    --os-disk-gb) OS_DISK_SIZE_GB="$2"; shift 2 ;;
    --repo-url) REPO_URL="$2"; shift 2 ;;
    --repo-ref) REPO_REF="$2"; shift 2 ;;
    --tags) TAGS="$2"; shift 2 ;;
    --bootstrap-log-path) BOOTSTRAP_LOG_PATH="$2"; shift 2 ;;
    --local-bootstrap-log-dir) LOCAL_BOOTSTRAP_LOG_DIR="$2"; shift 2 ;;
    --no-local-bootstrap-log) SAVE_LOCAL_BOOTSTRAP_LOG=0; shift ;;
    --no-bootstrap) AUTO_BOOTSTRAP=0; shift ;;
    --no-gpu-driver) INSTALL_GPU_DRIVER=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [ -z "${RESOURCE_GROUP}" ]; then
  echo "ERROR: --resource-group is required."
  usage
  exit 1
fi

if ! command -v az >/dev/null 2>&1; then
  echo "ERROR: az cli is not installed."
  exit 1
fi

if [ -z "${VM_NAME}" ]; then
  VM_NAME="probae-$(date +%Y%m%d-%H%M%S)"
fi
if [ -z "${BOOTSTRAP_LOG_PATH}" ]; then
  BOOTSTRAP_LOG_PATH="/home/${ADMIN_USERNAME}/probae_bootstrap.log"
fi

SSH_KEY_VALUE="${SSH_PUBLIC_KEY}"
if [ -f "${SSH_PUBLIC_KEY}" ]; then
  SSH_KEY_VALUE="$(cat "${SSH_PUBLIC_KEY}")"
fi
if ! [[ "${SSH_KEY_VALUE}" =~ ^(ssh-|ecdsa-|sk-) ]]; then
  echo "ERROR: --ssh-public-key must point to a public key file or contain key text."
  exit 1
fi

if [ -n "${SUBSCRIPTION_ID}" ]; then
  az account set --subscription "${SUBSCRIPTION_ID}"
fi

if az vm show --resource-group "${RESOURCE_GROUP}" --name "${VM_NAME}" >/dev/null 2>&1; then
  echo "ERROR: VM already exists: ${RESOURCE_GROUP}/${VM_NAME}"
  echo "Choose a different --name to ensure a fresh instance."
  exit 1
fi

if ! az group show --name "${RESOURCE_GROUP}" >/dev/null 2>&1; then
  echo "Creating resource group: ${RESOURCE_GROUP} (${LOCATION})"
  az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}" >/dev/null
fi

echo "Creating VM ..."
CREATE_ARGS=(
  vm create
  --resource-group "${RESOURCE_GROUP}"
  --name "${VM_NAME}"
  --location "${LOCATION}"
  --size "${VM_SIZE}"
  --image "${IMAGE_URN}"
  --admin-username "${ADMIN_USERNAME}"
  --ssh-key-values "${SSH_KEY_VALUE}"
  --public-ip-sku Standard
  --nsg-rule SSH
  --os-disk-size-gb "${OS_DISK_SIZE_GB}"
  -o json
)
if [ -n "${TAGS}" ]; then
  CREATE_ARGS+=(--tags "${TAGS}")
fi

CREATE_JSON="$(az "${CREATE_ARGS[@]}")"

PUBLIC_IP="$(
python3 - <<PY
import json
raw = """${CREATE_JSON}"""
d = json.loads(raw)
print(d.get("publicIpAddress", ""))
PY
)"

if [ "${INSTALL_GPU_DRIVER}" -eq 1 ]; then
  echo "Installing NVIDIA GPU driver extension ..."
  if az vm extension set \
    --resource-group "${RESOURCE_GROUP}" \
    --vm-name "${VM_NAME}" \
    --publisher Microsoft.HpcCompute \
    --name NvidiaGpuDriverLinux >/dev/null 2>&1; then
    echo "GPU driver extension installation triggered."
  else
    echo "WARN: NVIDIA driver extension install failed."
    echo "      You can retry manually with:"
    echo "      az vm extension set --resource-group ${RESOURCE_GROUP} --vm-name ${VM_NAME} --publisher Microsoft.HpcCompute --name NvidiaGpuDriverLinux"
  fi
fi

if [ "${AUTO_BOOTSTRAP}" -eq 1 ]; then
  echo "Bootstrapping VM with azure/setup_vm.sh (this can take several minutes) ..."
  BOOTSTRAP_SCRIPT="$(cat <<EOF
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

BOOTSTRAP_LOG_PATH="${BOOTSTRAP_LOG_PATH}"
mkdir -p "\$(dirname "\${BOOTSTRAP_LOG_PATH}")"
touch "\${BOOTSTRAP_LOG_PATH}"
exec >> "\${BOOTSTRAP_LOG_PATH}" 2>&1
echo "=== ProbAE bootstrap start: \$(date -Iseconds) ==="

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git python3 python3-pip python3-venv tmux
fi

REPO_DIR="/home/${ADMIN_USERNAME}/ProbAE_Deconv"
if [ ! -d "\${REPO_DIR}/.git" ]; then
  git clone "${REPO_URL}" "\${REPO_DIR}"
fi
cd "\${REPO_DIR}"

if [ -n "${REPO_REF}" ]; then
  git fetch --all --tags
  git checkout "${REPO_REF}"
fi

REPO_DIR="\${REPO_DIR}" REPO_URL="${REPO_URL}" REPO_REF="${REPO_REF}" VENV_DIR="\${REPO_DIR}/.venv" GIT_SYNC="ff-only" bash azure/setup_vm.sh
chown -R "${ADMIN_USERNAME}:${ADMIN_USERNAME}" "\${REPO_DIR}"
chown "${ADMIN_USERNAME}:${ADMIN_USERNAME}" "\${BOOTSTRAP_LOG_PATH}" || true
echo "=== ProbAE bootstrap done: \$(date -Iseconds) ==="
EOF
)"

  RUN_COMMAND_JSON="$(
  az vm run-command invoke \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VM_NAME}" \
    --command-id RunShellScript \
    --scripts "${BOOTSTRAP_SCRIPT}" \
    -o json
  )"

  LOCAL_BOOTSTRAP_LOG_FILE=""
  if [ "${SAVE_LOCAL_BOOTSTRAP_LOG}" -eq 1 ]; then
    mkdir -p "${LOCAL_BOOTSTRAP_LOG_DIR}"
    LOCAL_BOOTSTRAP_LOG_FILE="${LOCAL_BOOTSTRAP_LOG_DIR}/${VM_NAME}_bootstrap_$(date +%Y%m%d-%H%M%S).log"
    printf '%s' "${RUN_COMMAND_JSON}" | python3 - "${LOCAL_BOOTSTRAP_LOG_FILE}" <<'PY'
import json
import sys

out_path = sys.argv[1]
raw = sys.stdin.read().strip()

text = ""
try:
    doc = json.loads(raw) if raw else {}
    messages = []
    for item in doc.get("value", []):
        if isinstance(item, dict):
            msg = str(item.get("message", "")).strip()
            if msg:
                messages.append(msg)
    text = "\n\n".join(messages)
except Exception:
    text = raw

if not text:
    text = "(no output returned by az vm run-command invoke)"

with open(out_path, "w", encoding="utf-8") as handle:
    handle.write(text)
    if not text.endswith("\n"):
        handle.write("\n")
PY
  fi

  echo "Bootstrap completed."
fi

echo ""
echo "VM deployed successfully."
echo "  resource_group: ${RESOURCE_GROUP}"
echo "  vm_name:        ${VM_NAME}"
echo "  location:       ${LOCATION}"
echo "  size:           ${VM_SIZE}"
echo "  public_ip:      ${PUBLIC_IP}"
if [ "${AUTO_BOOTSTRAP}" -eq 1 ]; then
  echo "  vm_log:         ${BOOTSTRAP_LOG_PATH}"
  if [ "${SAVE_LOCAL_BOOTSTRAP_LOG}" -eq 1 ] && [ -n "${LOCAL_BOOTSTRAP_LOG_FILE}" ]; then
    echo "  local_log:      ${LOCAL_BOOTSTRAP_LOG_FILE}"
  fi
fi
echo ""
echo "Connect:"
echo "  ssh ${ADMIN_USERNAME}@${PUBLIC_IP}"
if [ "${AUTO_BOOTSTRAP}" -eq 1 ]; then
  echo "  tail -f ${BOOTSTRAP_LOG_PATH}"
fi
echo ""
echo "Run suite on VM:"
echo "  cd ~/ProbAE_Deconv"
echo "  source .venv/bin/activate"
echo "  bash runpod/run_suite.sh --config configs/experiment_suite.yaml --no-send --gpu-parallel auto"
echo ""
echo "Recommended long run (tmux):"
echo "  tmux new -s probae_suite"
echo "  cd ~/ProbAE_Deconv && source .venv/bin/activate"
echo "  bash runpod/run_suite.sh --config configs/experiment_suite.yaml --no-send --gpu-parallel auto"
echo ""
echo "Stop costs when done:"
echo "  az vm deallocate --resource-group ${RESOURCE_GROUP} --name ${VM_NAME}"
