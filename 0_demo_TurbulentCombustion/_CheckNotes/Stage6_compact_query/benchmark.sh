#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

package="_CheckNotes/Stage6_compact_query"
gpu_id="${GPU_ID:-2}"
free_mb="$(nvidia-smi -i "$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits)"
util_pct="$(nvidia-smi -i "$gpu_id" --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
if [[ "${FORCE_BUSY_GPU:-0}" != "1" && ( "$free_mb" -lt 30000 || "$util_pct" -gt 10 ) ]]; then
  echo "Refusing benchmark on busy GPU $gpu_id: ${free_mb} MiB free, ${util_pct}% utilization." >&2
  exit 3
fi

mkdir -p "$package/benchmarks" "$package/gpu_state"
timestamp="$(date +%Y%m%d_%H%M%S)"
nvidia-smi -i "$gpu_id"   --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu   --format=csv,noheader > "$package/gpu_state/benchmark_${timestamp}_initial.csv"

export CUDA_VISIBLE_DEVICES="$gpu_id"
export KEOPS_CACHE_FOLDER="/tmp/keops_stage6_cq_benchmark_gpu${gpu_id}"
conda run --no-capture-output -n phycoflow_env   python src/benchmark_pointcloud_cq.py   --device cuda:0   --query-sizes 4096 16384 65536   --n-obs 256   --iterations 5   --warmup 2   --component-iterations 5   --million-query-count 1000000   --million-chunk-size 8192   --output "$package/benchmarks/cost_benchmark.json"   2>&1 | tee "$package/benchmarks/cost_benchmark_${timestamp}.log"
