#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

package="_CheckNotes/Stage1_5_limited_run"
mkdir -p "$package/logs" "$package/runs"

export CUDA_VISIBLE_DEVICES=0
export KEOPS_CACHE_FOLDER=/tmp/keops_stage1_5_limited
nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "$package/gpu_initial_state.csv"

conda run --no-capture-output -n phycoflow_env \
  python src/train_pointcloud_ffm.py --config "$package/control.yaml" \
  2>&1 | tee "$package/logs/control.log"

control_run="$(ls -dt "$package"/runs/control_DemoN9201_* | head -1)"
conda run --no-capture-output -n phycoflow_env \
  python src/evaluate_pointcloud_fixed_manifest.py \
  --config "$package/control.yaml" \
  --manifest _CheckNotes/Stage1_fixed_val_manifest.pt \
  --checkpoint "$control_run/best.pt" \
  --device cuda:0 --batch-size 1 --repeats 1 --rf-seed 1729 \
  --output "$package/control_fixed_manifest.csv"

conda run --no-capture-output -n phycoflow_env \
  python src/train_pointcloud_ffm.py --config "$package/large_effective_query.yaml" \
  2>&1 | tee "$package/logs/large_effective_query.log"

large_run="$(ls -dt "$package"/runs/large_effective_query_DemoN9202_* | head -1)"
conda run --no-capture-output -n phycoflow_env \
  python src/evaluate_pointcloud_fixed_manifest.py \
  --config "$package/large_effective_query.yaml" \
  --manifest _CheckNotes/Stage1_fixed_val_manifest.pt \
  --checkpoint "$large_run/best.pt" \
  --device cuda:0 --batch-size 1 --repeats 1 --rf-seed 1729 \
  --output "$package/large_effective_query_fixed_manifest.csv"

printf '%s\n%s\n' "$control_run" "$large_run" > "$package/run_paths.txt"
python "$package/analyze.py"
