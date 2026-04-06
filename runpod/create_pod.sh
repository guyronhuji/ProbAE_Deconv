#!/usr/bin/env bash
# ============================================================
# Create a RunPod GPU pod for ProbAE_Deconv experiments.
#
# Prerequisites (one-time on local machine):
#   1. brew install runpod/runpodctl/runpodctl
#   2. runpodctl config set --apiKey <YOUR_KEY>
#      (or keep project .runpodkey with either:
#         apiKey <KEY>
#         <KEY>
#      )
#   3. Add SSH key at runpod.io -> Settings -> SSH Public Keys
#
# Usage:
#   bash runpod/create_pod.sh
#
# Optional env vars:
#   POD_NAME      (default: probae-deconv)
#   IMAGE         (default: runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04)
#   DISK_GB       (default: 80)
#   VOLUME_GB     (default: 30)
#   GPU_CHOICE    (non-interactive numeric choice)
# ============================================================

set -euo pipefail

POD_NAME="${POD_NAME:-probae-deconv}"
IMAGE="${IMAGE:-runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04}"
DISK_GB="${DISK_GB:-80}"
VOLUME_GB="${VOLUME_GB:-30}"
GPU_CHOICE="${GPU_CHOICE:-}"

if ! command -v runpodctl >/dev/null 2>&1; then
  echo "ERROR: runpodctl is not installed."
  echo "Install with: brew install runpod/runpodctl/runpodctl"
  exit 1
fi

KEYFILE="$(cd "$(dirname "$0")/.." && pwd)/.runpodkey"
RUNPOD_YAML="${HOME}/.runpod/.runpod.yaml"
API_KEY="${RUNPOD_API_KEY:-}"
API_KEY_SOURCE=""

extract_api_key() {
  local raw="$1"
  raw="$(echo "$raw" | tr -d '\r' | head -n1)"
  raw="${raw#apiKey}"
  raw="${raw#api_key}"
  raw="${raw#:}"
  raw="${raw#=}"
  raw="${raw# }"
  raw="${raw% }"
  echo "$raw"
}

if [ -z "$API_KEY" ] && [ -f "$KEYFILE" ]; then
  API_KEY="$(extract_api_key "$(cat "$KEYFILE")")"
  if [ -n "$API_KEY" ]; then
    API_KEY_SOURCE=".runpodkey"
  fi
fi
if [ -z "$API_KEY" ] && [ -f "$RUNPOD_YAML" ]; then
  API_KEY="$(awk -F': *' '/^apiKey:/ {print $2; exit}' "$RUNPOD_YAML" | tr -d '[:space:]')"
  if [ -n "$API_KEY" ]; then
    API_KEY_SOURCE="~/.runpod/.runpod.yaml"
  fi
fi
if [ -n "${RUNPOD_API_KEY:-}" ]; then
  API_KEY_SOURCE="RUNPOD_API_KEY"
fi

if [ -n "$API_KEY" ]; then
  echo "Using API key from ${API_KEY_SOURCE:-unknown source} for pricing query."
  mkdir -p "${HOME}/.runpod"
  printf "apiKey: %s\napiUrl: https://api.runpod.io/graphql\n" "$API_KEY" > "${HOME}/.runpod/.runpod.yaml"
fi

echo "Querying RunPod available GPU types ..."
GPU_RAW="$(runpodctl gpu list 2>/dev/null)" || {
  echo "ERROR: runpodctl gpu list failed"
  echo "Check authentication: runpodctl config set --apiKey <KEY>"
  exit 1
}

PRICE_JSON="{}"
if [ -n "$API_KEY" ]; then
  QUERY='{"query":"{ gpuTypes { id displayName lowestPrice(input: {gpuCount: 1}) { minimumBidPrice uninterruptablePrice } } }"}'
  PRICE_JSON="$(curl -sf -X POST "https://api.runpod.io/graphql?api_key=${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "$QUERY" || echo '{}')"
else
  echo "WARN: No API key found for GraphQL pricing; showing availability without price."
fi

echo "Probing availability counts ..."
PROBE_1="$(runpodctl get cloud 1 2>/dev/null | grep -v "^warning\|^GPU TYPE\|^$" || true)"
PROBE_2="$(runpodctl get cloud 2 2>/dev/null | grep -v "^warning\|^GPU TYPE\|^$" || true)"
PROBE_4="$(runpodctl get cloud 4 2>/dev/null | grep -v "^warning\|^GPU TYPE\|^$" || true)"
PROBE_8="$(runpodctl get cloud 8 2>/dev/null | grep -v "^warning\|^GPU TYPE\|^$" || true)"
PROBE_16="$(runpodctl get cloud 16 2>/dev/null | grep -v "^warning\|^GPU TYPE\|^$" || true)"

GPU_LIST="$(
GPU_RAW="$GPU_RAW" PRICE_JSON="$PRICE_JSON" \
PROBE_1="$PROBE_1" PROBE_2="$PROBE_2" PROBE_4="$PROBE_4" PROBE_8="$PROBE_8" PROBE_16="$PROBE_16" \
python3 - <<'PY'
import json
import os

gpu_list = json.loads(os.environ["GPU_RAW"])
price_raw = json.loads(os.environ.get("PRICE_JSON", "{}"))

prices = {}
for g in price_raw.get("data", {}).get("gpuTypes", []):
    lp = g.get("lowestPrice") or {}
    spot = lp.get("minimumBidPrice")
    od = lp.get("uninterruptablePrice")
    try:
        spot = float(spot) if spot is not None else None
    except Exception:
        spot = None
    try:
        od = float(od) if od is not None else None
    except Exception:
        od = None
    prices[g["id"]] = {
        "spot": f"${spot:.2f}" if isinstance(spot, (float, int)) else "-",
        "od": f"${od:.2f}" if isinstance(od, (float, int)) else "-",
    }

probes = {
    16: os.environ.get("PROBE_16", ""),
    8: os.environ.get("PROBE_8", ""),
    4: os.environ.get("PROBE_4", ""),
    2: os.environ.get("PROBE_2", ""),
    1: os.environ.get("PROBE_1", ""),
}

def max_avail(gpu_id: str) -> str:
    for n in [16, 8, 4, 2, 1]:
        if gpu_id in probes[n]:
            return f">={n}"
    return "?"

rows = []
for g in gpu_list:
    if not g.get("available"):
        continue
    gid = g["gpuId"]
    comm = bool(g.get("communityCloud", False))
    secure = bool(g.get("secureCloud", False))
    if comm and not secure:
        cloud = "COMMUNITY"
        cloud_label = "C"
    elif secure and not comm:
        cloud = "SECURE"
        cloud_label = "S"
    else:
        cloud = "COMMUNITY"
        cloud_label = "C+S"

    p = prices.get(gid, {"spot": "-", "od": "-"})
    rows.append({
        "id": gid,
        "name": g.get("displayName", gid),
        "vram": g.get("memoryInGb", "?"),
        "spot": p["spot"],
        "od": p["od"],
        "avail": max_avail(gid),
        "cloud": cloud,
        "cloudl": cloud_label,
    })

def sort_od(x):
    try:
        return float(str(x["od"]).strip("$"))
    except Exception:
        return 1e9

rows.sort(key=sort_od)
print(json.dumps(rows))
PY
)"

if [ -z "$GPU_LIST" ] || [ "$GPU_LIST" = "[]" ]; then
  echo "No GPUs currently available. Try again in a few minutes."
  exit 1
fi

RUNPOD_GPU_LIST="$GPU_LIST" python3 - <<'PY'
import json
import os
rows = json.loads(os.environ["RUNPOD_GPU_LIST"])
print("")
print(f"{'#':<4} {'GPU':<30} {'VRAM':>7} {'SPOT $/hr':>10} {'OD $/hr':>9} {'AVAIL':>7} {'CLOUD':>6}")
print(f"{'---':<4} {'-'*30} {'-'*7:>7} {'-'*10:>10} {'-'*9:>9} {'-'*7:>7} {'-'*6:>6}")
for i, r in enumerate(rows, start=1):
    print(f"{i:<4} {r['name']:<30} {str(r['vram'])+'GB':>7} {r['spot']:>10} {r['od']:>9} {r['avail']:>7} {r['cloudl']:>6}")
PY

eval "$(
RUNPOD_GPU_LIST="$GPU_LIST" python3 - <<'PY'
import json
import os
rows = json.loads(os.environ["RUNPOD_GPU_LIST"])
ids = " ".join(f'"{x["id"]}"' for x in rows)
names = " ".join(f'"{x["name"]}"' for x in rows)
clouds = " ".join(f'"{x["cloud"]}"' for x in rows)
print(f"GPU_IDS=({ids})")
print(f"GPU_NAMES=({names})")
print(f"GPU_CLOUDS=({clouds})")
print(f"TOTAL={len(rows)}")
PY
)"

if [ -z "$GPU_CHOICE" ]; then
  echo ""
  read -rp "Choose GPU number (1-${TOTAL}): " GPU_CHOICE
fi

if ! [[ "$GPU_CHOICE" =~ ^[0-9]+$ ]] || [ "$GPU_CHOICE" -lt 1 ] || [ "$GPU_CHOICE" -gt "$TOTAL" ]; then
  echo "Invalid GPU choice: ${GPU_CHOICE}"
  exit 1
fi

IDX=$((GPU_CHOICE - 1))
SELECTED_ID="${GPU_IDS[$IDX]}"
SELECTED_NAME="${GPU_NAMES[$IDX]}"
SELECTED_CLOUD="${GPU_CLOUDS[$IDX]}"

echo ""
echo "Creating pod ..."
echo "  Name : ${POD_NAME}"
echo "  GPU  : ${SELECTED_NAME}"
echo "  Cloud: ${SELECTED_CLOUD}"
echo "  Image: ${IMAGE}"

OUTPUT="$(runpodctl pod create \
  --name "$POD_NAME" \
  --gpu-id "$SELECTED_ID" \
  --image "$IMAGE" \
  --container-disk-in-gb "$DISK_GB" \
  --volume-in-gb "$VOLUME_GB" \
  --volume-mount-path "/workspace" \
  --cloud-type "$SELECTED_CLOUD" \
  --ssh \
  2>&1)" && {
    POD_ID="$(
      echo "$OUTPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id',''))" 2>/dev/null || \
      echo "$OUTPUT" | grep -oE '"id":"[^"]*"' | head -1 | cut -d'"' -f4 || true
    )"
    echo ""
    echo "============================================================"
    echo "Pod created."
    [ -n "$POD_ID" ] && echo "Pod ID: $POD_ID"
    echo ""
    echo "Next steps:"
    echo "1) Open: https://www.runpod.io/console/pods"
    echo "2) Connect via SSH"
    echo "3) In pod:"
    echo "     cd /workspace"
    echo "     git clone https://github.com/guyronhuji/ProbAE_Deconv.git"
    echo "     cd ProbAE_Deconv"
    echo "     bash runpod/setup_pod.sh"
    echo ""
    echo "Stop pod when done:"
    if [ -n "$POD_ID" ]; then
      echo "  runpodctl stop pod $POD_ID"
    else
      echo "  runpodctl pod list  # then stop by ID"
    fi
    echo "============================================================"
  } || {
    echo "ERROR: pod creation failed"
    echo "$OUTPUT"
    exit 1
  }
