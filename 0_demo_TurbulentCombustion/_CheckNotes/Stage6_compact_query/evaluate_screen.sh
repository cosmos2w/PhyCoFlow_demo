#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

package="_CheckNotes/Stage6_compact_query"
candidate="${1:-}"
gpu_id="${GPU_ID:-2}"

case "$candidate" in
  full)
    screen_dir="$package/screen_cq_full"
    run_prefix="CQ_Full"
    ;;
  lr)
    screen_dir="$package/screen_cq_lr"
    run_prefix="CQ_LR"
    ;;
  rescue160)
    screen_dir="$package/screen_cq_rescue160"
    run_prefix="CQ_Rescue160"
    ;;
  *)
    echo "Usage: GPU_ID=<physical-gpu> bash $package/evaluate_screen.sh full|lr|rescue160 [run-dir]" >&2
    exit 2
    ;;
esac

run_dir="${2:-}"
if [[ -z "$run_dir" ]]; then
  run_dir="$(ls -dt "$screen_dir/runs/${run_prefix}_DemoN"* | head -1)"
fi
for epoch in 0001 0020 0040 0060; do
  test -f "$run_dir/epoch_${epoch}.pt"
done

free_mb="$(nvidia-smi -i "$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits)"
util_pct="$(nvidia-smi -i "$gpu_id" --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
if [[ "${FORCE_BUSY_GPU:-0}" != "1" && ( "$free_mb" -lt 12000 || "$util_pct" -gt 20 ) ]]; then
  echo "Refusing evaluation on busy GPU $gpu_id: ${free_mb} MiB free, ${util_pct}% utilization." >&2
  exit 3
fi

mkdir -p "$screen_dir/evaluation/fixed_manifest" "$screen_dir/evaluation/matched_reconstruction"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export KEOPS_CACHE_FOLDER="/tmp/keops_stage6_cq_eval_gpu${gpu_id}"

conda run --no-capture-output -n phycoflow_env python src/evaluate_pointcloud_fixed_manifest.py \
  --config "$run_dir/run_config.yaml" \
  --manifest _CheckNotes/Stage1_fixed_val_manifest.pt \
  --checkpoint "$run_dir/epoch_0001.pt" "$run_dir/epoch_0020.pt" "$run_dir/epoch_0040.pt" "$run_dir/epoch_0060.pt" \
  --device cuda:0 --batch-size 1 --repeats 3 --rf-seed 1729 \
  --output "$screen_dir/evaluation/fixed_manifest/milestones.json"

conda run --no-capture-output -n phycoflow_env python _CheckNotes/Stage6_formal_baseline/evaluate_matched_reconstruction.py \
  --run "$run_dir" --checkpoint epoch_0060.pt \
  --output "$screen_dir/evaluation/matched_reconstruction/epoch_0060" \
  --device cuda:0 --condition-seed 42 --sample-seed 1042 --nfe 1 2 4
