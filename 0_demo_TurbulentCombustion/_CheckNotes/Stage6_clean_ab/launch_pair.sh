#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

package="_CheckNotes/Stage6_clean_ab"
old_gpu="${OLD_GPU:-0}"
new_gpu="${NEW_GPU:-1}"

if [[ "$old_gpu" == "$new_gpu" ]]; then
  echo "OLD_GPU and NEW_GPU must be different physical GPUs." >&2
  exit 2
fi

conda run --no-capture-output -n phycoflow_env python "$package/validate_configs.py"

for gpu in "$old_gpu" "$new_gpu"; do
  free_mb="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
  util_pct="$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
  if [[ "${FORCE_BUSY_GPU:-0}" != "1" && ( "$free_mb" -lt 40000 || "$util_pct" -gt 10 ) ]]; then
    echo "Refusing GPU $gpu: ${free_mb} MiB free, ${util_pct}% utilization." >&2
    exit 3
  fi
done

mkdir -p "$package/logs" "$package/gpu_state" "$package/runs"
timestamp="$(date +%Y%m%d_%H%M%S)"
nvidia-smi -i "$old_gpu" --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$package/gpu_state/F0_ENH_${timestamp}_initial.csv"
nvidia-smi -i "$new_gpu" --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$package/gpu_state/CQ_LR_${timestamp}_initial.csv"

old_log="$package/logs/F0_ENH_${timestamp}.log"
new_log="$package/logs/CQ_LR_${timestamp}.log"

echo "Launching F0-ENH on physical GPU $old_gpu -> $old_log"
CUDA_VISIBLE_DEVICES="$old_gpu" KEOPS_CACHE_FOLDER="/tmp/keops_stage6_clean_ab_old_gpu${old_gpu}" \
  conda run --no-capture-output -n phycoflow_env \
  python src/train_pointcloud_ffm.py --config "$package/F0_ENH_60ep.yaml" \
  > "$old_log" 2>&1 &
old_pid=$!

echo "Launching CQ-LR on physical GPU $new_gpu -> $new_log"
CUDA_VISIBLE_DEVICES="$new_gpu" KEOPS_CACHE_FOLDER="/tmp/keops_stage6_clean_ab_new_gpu${new_gpu}" \
  conda run --no-capture-output -n phycoflow_env \
  python src/train_pointcloud_ffm.py --config "$package/CQ_LR_60ep.yaml" \
  > "$new_log" 2>&1 &
new_pid=$!

printf 'timestamp=%s\nold_gpu=%s\nnew_gpu=%s\nold_pid=%s\nnew_pid=%s\nold_log=%s\nnew_log=%s\n' \
  "$timestamp" "$old_gpu" "$new_gpu" "$old_pid" "$new_pid" "$old_log" "$new_log" \
  > "$package/active_pair.env"

echo "PIDs: F0-ENH=$old_pid, CQ-LR=$new_pid"
echo "Monitor: tail -f $old_log $new_log"

set +e
wait "$old_pid"
old_status=$?
wait "$new_pid"
new_status=$?
set -e

if [[ "$old_status" -ne 0 || "$new_status" -ne 0 ]]; then
  echo "Training failure: F0-ENH=$old_status, CQ-LR=$new_status" >&2
  exit 4
fi

old_run="$(ls -dt "$package/runs/F0_ENH_DemoN9500_"* | head -1)"
new_run="$(ls -dt "$package/runs/CQ_LR_DemoN9501_"* | head -1)"
echo "Training complete. Running controlled fixed-manifest evaluation."
OLD_GPU="$old_gpu" NEW_GPU="$new_gpu" \
  bash "$package/evaluate_pair.sh" "$old_run" "$new_run"

conda run --no-capture-output -n phycoflow_env \
  python "$package/analyze_pair.py" --baseline-run "$old_run" --new-run "$new_run"

echo "Complete: $package/RESULTS.md"
