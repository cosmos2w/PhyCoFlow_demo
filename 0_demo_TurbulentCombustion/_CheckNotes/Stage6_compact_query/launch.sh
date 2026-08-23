#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

package="_CheckNotes/Stage6_compact_query"
candidate="${1:-}"
gpu_id="${GPU_ID:-2}"

case "$candidate" in
  full)
    config="$package/CQ_full_60ep.yaml"
    run_label="CQ_Full"
    ;;
  lr)
    config="$package/CQ_lr_60ep.yaml"
    run_label="CQ_LR"
    ;;
  rescue160)
    config="$package/CQ_rescue160_60ep.yaml"
    run_label="CQ_Rescue160"
    ;;
  *)
    echo "Usage: GPU_ID=<physical-gpu> bash $package/launch.sh full|lr|rescue160" >&2
    exit 2
    ;;
esac

conda run --no-capture-output -n phycoflow_env python "$package/validate_configs.py"

free_mb="$(nvidia-smi -i "$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits)"
util_pct="$(nvidia-smi -i "$gpu_id" --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
if [[ "${FORCE_BUSY_GPU:-0}" != "1" && ( "$free_mb" -lt 30000 || "$util_pct" -gt 10 ) ]]; then
  echo "Refusing to launch on busy GPU $gpu_id: ${free_mb} MiB free, ${util_pct}% utilization." >&2
  exit 3
fi

mkdir -p "$package/logs" "$package/gpu_state"
timestamp="$(date +%Y%m%d_%H%M%S)"
nvidia-smi -i "$gpu_id" --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$package/gpu_state/${run_label}_${timestamp}_initial.csv"

export CUDA_VISIBLE_DEVICES="$gpu_id"
export KEOPS_CACHE_FOLDER="/tmp/keops_stage6_cq_gpu${gpu_id}"

conda run --no-capture-output -n phycoflow_env python src/train_pointcloud_ffm.py --config "$config" 2>&1 | tee "$package/logs/${run_label}_${timestamp}.log"
