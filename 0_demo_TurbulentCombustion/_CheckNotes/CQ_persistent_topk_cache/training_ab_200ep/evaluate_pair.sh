#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

package="_CheckNotes/CQ_persistent_topk_cache/training_ab_200ep"
old_gpu="${OLD_GPU:-0}"
new_gpu="${NEW_GPU:-1}"
current_project="${CURRENT_PROJECT:-$repo_root}"
old_run="${1:?baseline run directory is required}"
new_run="${2:?persistent run directory is required}"
milestones=(0001 0020 0040 0060 0100 0150 0200)

old_checkpoints=()
new_checkpoints=()
for epoch in "${milestones[@]}"; do
  old_checkpoints+=("$old_run/epoch_${epoch}.pt")
  new_checkpoints+=("$new_run/epoch_${epoch}.pt")
done
for checkpoint in "${old_checkpoints[@]}" "${new_checkpoints[@]}"; do
  test -f "$checkpoint"
done

mkdir -p "$package/evaluation/no_persistent" \
  "$package/evaluation/persistent_topk" "$package/benchmarks" "$package/logs"

printf 'stage=fixed_manifest\n' > "$package/pipeline_status.env"
CUDA_VISIBLE_DEVICES="$old_gpu" KEOPS_CACHE_FOLDER="/tmp/keops_cq_topk_ab_eval_old_gpu${old_gpu}" \
  conda run --no-capture-output -n phycoflow_env \
  python "$current_project/src/evaluate_pointcloud_fixed_manifest.py" \
  --config "$old_run/run_config.yaml" \
  --manifest _CheckNotes/Stage1_fixed_val_manifest.pt \
  --checkpoint "${old_checkpoints[@]}" \
  --device cuda:0 --batch-size 1 --repeats 3 --rf-seed 1729 \
  --output "$package/evaluation/no_persistent/milestones.json" \
  > "$package/logs/no_persistent_fixed_manifest.log" 2>&1 &
old_eval_pid=$!

CUDA_VISIBLE_DEVICES="$new_gpu" KEOPS_CACHE_FOLDER="/tmp/keops_cq_topk_ab_eval_new_gpu${new_gpu}" \
  conda run --no-capture-output -n phycoflow_env \
  python "$current_project/src/evaluate_pointcloud_fixed_manifest.py" \
  --config "$new_run/run_config.yaml" \
  --manifest _CheckNotes/Stage1_fixed_val_manifest.pt \
  --checkpoint "${new_checkpoints[@]}" \
  --device cuda:0 --batch-size 1 --repeats 3 --rf-seed 1729 \
  --output "$package/evaluation/persistent_topk/milestones.json" \
  > "$package/logs/persistent_topk_fixed_manifest.log" 2>&1 &
new_eval_pid=$!

wait "$old_eval_pid"
wait "$new_eval_pid"

printf 'stage=reconstruction_benchmark\n' > "$package/pipeline_status.env"
stats="Save_TrainedModel/ffm_tc_pointcloud_DemoN51_20260718_083538/dataset_stats.pt"
CUDA_VISIBLE_DEVICES="$old_gpu" KEOPS_CACHE_FOLDER="/tmp/keops_cq_topk_ab_bench_old_gpu${old_gpu}" \
  conda run --no-capture-output -n phycoflow_env \
  python "$current_project/research/benchmarks/benchmark_cq_persistent_topk_cache.py" \
  --config "$old_run/run_config.yaml" --checkpoint "$old_run/epoch_0200.pt" \
  --stats-path "$stats" --device cuda:0 --n-points 250000 1000000 \
  --nfe 1 2 4 8 --n-obs 256 --chunk-size 8192 --repeats 3 --warmup-repeats 1 \
  --output-csv "$package/benchmarks/no_persistent_checkpoint.csv" \
  --output-json "$package/benchmarks/no_persistent_checkpoint.json" \
  > "$package/logs/no_persistent_benchmark.log" 2>&1 &
old_bench_pid=$!

CUDA_VISIBLE_DEVICES="$new_gpu" KEOPS_CACHE_FOLDER="/tmp/keops_cq_topk_ab_bench_new_gpu${new_gpu}" \
  conda run --no-capture-output -n phycoflow_env \
  python "$current_project/research/benchmarks/benchmark_cq_persistent_topk_cache.py" \
  --config "$new_run/run_config.yaml" --checkpoint "$new_run/epoch_0200.pt" \
  --stats-path "$stats" --device cuda:0 --n-points 250000 1000000 \
  --nfe 1 2 4 8 --n-obs 256 --chunk-size 8192 --repeats 3 --warmup-repeats 1 \
  --output-csv "$package/benchmarks/persistent_topk_checkpoint.csv" \
  --output-json "$package/benchmarks/persistent_topk_checkpoint.json" \
  > "$package/logs/persistent_topk_benchmark.log" 2>&1 &
new_bench_pid=$!

wait "$old_bench_pid"
wait "$new_bench_pid"
echo "Evaluation and reconstruction benchmarks complete."
