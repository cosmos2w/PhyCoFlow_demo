#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

package="_CheckNotes/Stage6_compact_query"
config="$package/formal_candidate/selected_200ep.yaml"
gpu_id="${GPU_ID:-2}"

if [[ "${ALLOW_STAGE6_FORMAL_RUN:-0}" != "1" ]]; then
  echo "Formal Stage-6 launch is disabled. Set ALLOW_STAGE6_FORMAL_RUN=1 only after explicit approval." >&2
  exit 2
fi
test -f "$package/formal_candidate/selection.json"
test -f "$config"

free_mb="$(nvidia-smi -i "$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits)"
util_pct="$(nvidia-smi -i "$gpu_id" --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
if [[ "$free_mb" -lt 30000 || "$util_pct" -gt 10 ]]; then
  echo "Refusing formal run on busy GPU $gpu_id: ${free_mb} MiB free, ${util_pct}% utilization." >&2
  exit 3
fi

mkdir -p "$package/formal_candidate/logs" "$package/gpu_state"
timestamp="$(date +%Y%m%d_%H%M%S)"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export KEOPS_CACHE_FOLDER="/tmp/keops_stage6_cq_formal_gpu${gpu_id}"
conda run --no-capture-output -n phycoflow_env   python src/train_pointcloud_ffm.py --config "$config"   2>&1 | tee "$package/formal_candidate/logs/formal_${timestamp}.log"
