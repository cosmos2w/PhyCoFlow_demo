#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

package="_CheckNotes/Stage6_clean_ab"
old_gpu="${OLD_GPU:-0}"
new_gpu="${NEW_GPU:-1}"
old_run="${1:-}"
new_run="${2:-}"

if [[ -z "$old_run" ]]; then
  old_run="$(ls -dt "$package/runs/F0_ENH_DemoN9500_"* | head -1)"
fi
if [[ -z "$new_run" ]]; then
  new_run="$(ls -dt "$package/runs/CQ_LR_DemoN9501_"* | head -1)"
fi

for run in "$old_run" "$new_run"; do
  for epoch in 0001 0020 0040 0060; do
    test -f "$run/epoch_${epoch}.pt"
  done
done

mkdir -p "$package/evaluation/F0_ENH" "$package/evaluation/CQ_LR" "$package/logs"

CUDA_VISIBLE_DEVICES="$old_gpu" KEOPS_CACHE_FOLDER="/tmp/keops_stage6_clean_ab_eval_old_gpu${old_gpu}" \
  conda run --no-capture-output -n phycoflow_env \
  python src/evaluate_pointcloud_fixed_manifest.py \
  --config "$old_run/run_config.yaml" \
  --manifest _CheckNotes/Stage1_fixed_val_manifest.pt \
  --checkpoint "$old_run/epoch_0001.pt" "$old_run/epoch_0020.pt" "$old_run/epoch_0040.pt" "$old_run/epoch_0060.pt" \
  --device cuda:0 --batch-size 1 --repeats 3 --rf-seed 1729 \
  --output "$package/evaluation/F0_ENH/milestones.json" \
  > "$package/logs/F0_ENH_fixed_manifest.log" 2>&1 &
old_pid=$!

CUDA_VISIBLE_DEVICES="$new_gpu" KEOPS_CACHE_FOLDER="/tmp/keops_stage6_clean_ab_eval_new_gpu${new_gpu}" \
  conda run --no-capture-output -n phycoflow_env \
  python src/evaluate_pointcloud_fixed_manifest.py \
  --config "$new_run/run_config.yaml" \
  --manifest _CheckNotes/Stage1_fixed_val_manifest.pt \
  --checkpoint "$new_run/epoch_0001.pt" "$new_run/epoch_0020.pt" "$new_run/epoch_0040.pt" "$new_run/epoch_0060.pt" \
  --device cuda:0 --batch-size 1 --repeats 3 --rf-seed 1729 \
  --output "$package/evaluation/CQ_LR/milestones.json" \
  > "$package/logs/CQ_LR_fixed_manifest.log" 2>&1 &
new_pid=$!

wait "$old_pid"
wait "$new_pid"
echo "Fixed-manifest evaluation complete."
